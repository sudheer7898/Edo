import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import time
import cv2
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

import utils.constants as c
import json
import torchvision.transforms.functional as TF
import sys

from baseModel import ptm,Embedding
from transformerBlock import (
    temporalAttentionPooling,
    TransformerBlock
)
from neck import FeatureAggregatorToTransformer
from fvcore.nn import parameter_count_table

save_interval:int=c.save_interval

class ClassifierHead(nn.Module):
    def __init__(self,dim_in,dim_out,dropout=0.3):
        super().__init__()
        self.nn=nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Dropout(dropout),
            nn.Linear(dim_in,dim_in//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128,dim_out)
        )
    def forward(self,x):
        return self.nn(x)

class Model(nn.Module):
    def __init__(self, num_blocks=c.transformer_blocks, num_frames=c.num_frames):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_frames = num_frames
        self.preTrained = ptm()
        self.in_channels = [256, 512, 1024, 2048]
        self.BiFPN = FeatureAggregatorToTransformer(
            in_channels_list=self.in_channels, d_model=256, target_size=(80, 80)
        )
        print("Initialized BiFPN")
        self.Embedding = Embedding(img_Size=80, patch_size=5, n_channels=256, out_dim=256)
        print("Initialized Embedding Layer")
        self.Transformer = nn.ModuleList([
            TransformerBlock(d_model=256, num_frames=self.num_frames, num_patches=256, num_heads=4) 
            for _ in range(self.num_blocks)
        ])
        print(f"Initialized {self.num_blocks} Transformers")
        self.poolingLayer = temporalAttentionPooling(256, 256, 4)
        self.pool_proj=nn.Linear(256,256)
        print("Initialized pooling Layer")
        self.classifierHead = ClassifierHead(256, 3)
        for param in self.preTrained.parameters():
            param.requires_grad = False
        for name, param in self.preTrained.named_parameters():
            if "blocks.5" in name or "blocks.6" in name or "conv_head" in name:
                param.requires_grad = True

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F == self.num_frames, f"Expected {self.num_frames} but received {F}"
        x = x.view(B * F, C, H, W)
        featureMapList = self.preTrained(x)
        featureMap = self.BiFPN(featureMapList)
        featureMap = featureMap.permute(0, 3, 1, 2)
        embed = self.Embedding(featureMap)
        y = embed
        for block in self.Transformer:
            y = block(y)
        cls_token = y[:, 0, :]
        patch_tokens = y[:, 1:, :]
        pooling_input = patch_tokens.view(B, F, 16, 16, 256)
        pooled_matrix = self.poolingLayer(pooling_input)   
        spatial_gap = pooled_matrix.mean(dim=-1)             
        spatial_gap = self.pool_proj(spatial_gap)            
        class_logits =  self.classifierHead(spatial_gap)  
        return class_logits

choices = ["ColorJitter", "GrayScale", "RandomAffine"]

def applyAugmentation(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies consistent sequence-wide transformations directly to the 4D video tensor.
    Modifies and returns a transformed copy.
    """
    choice = np.random.choice(choices)
    b = np.random.uniform(0.5, 1.5)
    c = np.random.uniform(0.5, 1.5)
    hue_val = np.random.uniform(-0.1, 0.1)  
    d = float(np.random.uniform(-10, 10))
    for t in range(video_tensor.size(0)):
        img = video_tensor[t]
        if choice == "ColorJitter":
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_hue(img, hue_val)
        elif choice == "GrayScale":
            gray = TF.rgb_to_grayscale(img)
            img = torch.cat([gray, gray, gray], dim=0)
        elif choice == "RandomAffine":
            _, img_h, img_w = img.shape
            img = TF.affine(img, d, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], center=[img_w // 2, img_h // 2])
        video_tensor[t] = img
    return video_tensor

class CrackImageToVideoDataset(Dataset):
    def __init__(self, DatasetPath_list=[
                     r"data\CrackDatatrain",
                     r"data\Nocrack",
                     r"data\passiveCrack",
                     r"data\passive-2"
                 ], shuffle=False, input_size=640) -> None:
        super().__init__()    
        self.input_size = input_size
        base_records = []
        for path in DatasetPath_list:
            csv_dir = os.path.join(path, "td-1")
            if not os.path.exists(csv_dir):
                continue
            addresses = os.listdir(csv_dir)
            for csv_file in addresses:
                base_records.append({
                    "csv_file": csv_file,
                    "csv_dir": csv_dir,
                    "is_aug_duplicate": False 
                })
        df_base = pd.DataFrame(base_records)
        duplicated_records = []
        for record in base_records:
            if "Nocrack" not in record["csv_dir"]:
                dup = record.copy()
                dup["is_aug_duplicate"] = True  
                duplicated_records.append(dup)
        if len(duplicated_records) > 0:
            df_aug = pd.DataFrame(duplicated_records)
            self.df = pd.concat([df_base, df_aug], ignore_index=True)
        else:
            self.df = df_base
        dup_df = self.df.copy()
        dup_df["is_aug_duplicate"] = True
        self.df = pd.concat([self.df, dup_df], ignore_index=True)
        if shuffle and not self.df.empty:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        print(f"Dataset initialization successful.")
        print(f"-> Base samples: {len(df_base)} | Added minority synthetic duplicates: {len(duplicated_records)}")
        print(f"-> Total sequence windows loaded: {len(self.df)}")
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row_data = self.df.iloc[index]
        csv_file = row_data["csv_file"]
        csv_dir = row_data["csv_dir"]
        is_aug_duplicate = row_data["is_aug_duplicate"]
        path = os.path.join(csv_dir, csv_file)
        tempDf = pd.read_csv(path)
        img_tensors = []
        for _, row in tempDf.iterrows():
            abs_img_path = row["img"]
            img = cv2.imread(abs_img_path)
            if img is None:
                raise FileNotFoundError(f"The image {abs_img_path} cannot be read.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = TF.resize(img_tensor, [self.input_size, self.input_size])
            img_tensors.append(img_tensor)
        video_tensor = torch.stack(img_tensors)
        label_idx = int(tempDf["label"].iloc[0]) if "label" in tempDf.columns else 0
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        if is_aug_duplicate:
            video_tensor = applyAugmentation(video_tensor)
        return video_tensor, label_tensor
    
def save(model: nn.Module, file: str = r"results\finalModel.pth"):
    os.makedirs(r"results",exist_ok=True)
    torch.save(model.state_dict(), file)

def load(model: nn.Module, file: str = r"results\finalModel.pth"):
    if os.path.exists(file):
        state_dict = torch.load(file, map_location=lambda storage, loc: storage, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint weights from {file}")
    else:
        print(f" Checkpoint {file} not found. Starting from scratch.")

def saveStatus(epoch: int, loss: float, rem_epochs: int = 0):
    status = {
        "epoch": epoch,
        "loss": loss,
        "remainingEpochs": rem_epochs
    }
    with open("status.json", "w") as file:
        json.dump(status, file, indent=4)
    print("Saved current training metrics status to status.json")

total_loss_train = []

val_acc_history  = [0,0,0]

def train(isPreviouslyTrained: bool = True, isNewData: bool = False):
    st_t = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Target GPU Accelerator:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print(f"Active Runtime Device context: {str(device)}\n")
    model = Model(num_blocks=c.transformer_blocks, num_frames=c.num_frames).to(device)
    epochs = c.epochs
    if isPreviouslyTrained:
        load(model,file=r"resultsBest\bestAccModel.pth")
        if not isNewData:
            if os.path.exists("status.json"):
                with open("status.json", "r") as f:
                    statusData = json.load(f)
                    epochs = statusData.get("remainingEpochs", epochs)
    backbone_params = [p for p in model.preTrained.parameters() if p.requires_grad]
    head_params     = [p for p in model.parameters()
                       if p.requires_grad and not any(p is bp for bp in model.preTrained.parameters())]
    # FIX: SGD + Nesterov momentum — far more stable than AdamW on tiny batches
    # optimizer = torch.optim.SGD([
    #     {'params': backbone_params, 'lr': 1e-5},
    #     {'params': head_params,     'lr': 1e-3}
    # ], momentum=0.9, weight_decay=1e-4)
    optimizer=torch.optim.AdamW([
        {'params':backbone_params, 'lr':1e-5},
        {"params":model.BiFPN.parameters(),"lr":5e-5},
        {"params":model.Embedding.parameters(),"lr":5e-5},
        {"params":model.Transformer.parameters(),"lr":5e-5},
        {"params":model.poolingLayer.parameters(),"lr":1e-5},
        {"params":model.classifierHead.parameters(),"lr":1e-4},
        ],weight_decay=1e-4)
    # optimizer=torch.optim.AdamW([
    #     {'params':backbone_params, 'lr':1e-6},
    #     {"params":model.BiFPN.parameters(),"lr":5e-6},
    #     {"params":model.Embedding.parameters(),"lr":5e-6},
    #     {"params":model.Transformer.parameters(),"lr":5e-6},
    #     {"params":model.poolingLayer.parameters(),"lr":1e-6},
    #     {"params":model.classifierHead.parameters(),"lr":1e-5},
    #     ],weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=c.epochs,eta_min=1e-6
    )
    weights=torch.tensor([1.0,2.5,2.5],dtype=torch.float32).to(device)
    classification_loss_fn = nn.CrossEntropyLoss(weight=weights)
    dataset   = CrackImageToVideoDataset()
    dataloader = DataLoader(
        dataset, batch_size=c.batch_size, shuffle=True,
        num_workers=2
    )
    val_dataset    = CrackImageToVideoDataset(DatasetPath_list=[c.ValDatsetPath])
    val_dataloader = DataLoader(
        val_dataset, batch_size=c.batch_size, shuffle=True,
        num_workers=2
    )
    print(f"Validation dataset loaded: {len(val_dataset)} items")
    ed_t = time.perf_counter()
    print(f"time req for dataset:{(ed_t - st_t):.2f}")
    best_val_loss    = 0
    patience         = 15
    patience_counter = 0
    for epoch in range(epochs):
        print(f"\n Execution Started: Training started on {len(dataset)} items...")
        start_time = time.perf_counter()
        epoch_loss_total = 0.0
        model.train()
        for batch_idx, (inputs, target_class) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs       = inputs.to(device)
            target_class = target_class.to(device)      
            class_logits = model(inputs)
            loss_class = classification_loss_fn(class_logits, target_class)
            ##################################################
            total_loss = loss_class 
            ##################################################
            total_loss.backward()
            optimizer.step()
            epoch_loss_total += total_loss.item()
            tim = time.perf_counter()
            sys.stdout.write(
                f"\rEpoch: {epoch+1} | Batch: {batch_idx + 1}/{len(dataloader)} |"
                f"Progress: {((batch_idx + 1) / len(dataloader) * 100):.1f}% | "
                f"Time: {(tim - start_time):.2f}s"
            )
            sys.stdout.flush()
        end_time   = time.perf_counter()
        num_batches = len(dataloader) if len(dataloader) > 0 else 1
        avg_loss    = epoch_loss_total / num_batches
        print(
            f"\n Epoch: {epoch + 1}/{epochs} Finished | "
            f"Avg Batch Loss: {avg_loss:.4f} | "
            f"Time: {end_time - start_time:.2f}s"
        )
        total_loss_train.append(epoch_loss_total)
        val(model=model, device=device, val_dataloader=val_dataloader)
        curr_val_loss = total_loss_val[-1]
        scheduler.step()
        if (epoch + 1) % save_interval == 0:
            print("Saving model state checkpoint to finalModel.pth...")
            save(model)
            saveStatus(epoch=epoch + 1, loss=epoch_loss_total, rem_epochs=epochs - (epoch + 1))
            print("-" * 50)
        if epoch == epochs - 1:
            save(model)
            print(f"saved model to finalModel.pth")
        temp=val_acc_history[:-1]
        curr_acc=val_acc_history[-1]
        if not temp or curr_acc > max(temp):
            print(f"accuracy increased to {curr_acc:.4f}")
            save(model,file=r"results\bestAccModel.pth")
            print(r"saved model to results\bestAccModel.pth")
        if curr_val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss:.4f} to {curr_val_loss:.4f}")
            print("saving best checkpoint")
            save(model,file=r"results\bestModel.pth")
            best_val_loss    = curr_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"\nValidation loss did not improve. "
                f"Early stopping counter: {patience_counter}/{patience}"
            )
            if patience_counter > patience:
                print(f"Early stopping triggered. Training stopped at epoch: {epoch + 1}.")
                break
        print(f"total loss Train:{epoch_loss_total}")
    print("saving plots")
    drawPlots()
    ed_t = time.perf_counter()
    print(f"Total time taken for training = {ed_t - st_t:.2f}s")
    print("=========================================================================")
    print(" Success: Multi-task gradient flow tracking validation completed!")
    print("=========================================================================")

total_loss_val  = []

def val(model: Model, device: torch.device, val_dataloader: DataLoader):
    global total_loss_val, val_acc_history
    classification_loss_fn = nn.CrossEntropyLoss()
    print(f"\n Execution Started: Validation started on {len(val_dataloader)} items...")
    start_time = time.perf_counter()
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch_idx, (inputs, target_class) in enumerate(val_dataloader):
            inputs       = inputs.to(device)
            target_class = target_class.to(device)   
            class_logits = model(inputs)
            batch_class = classification_loss_fn(class_logits, target_class)
            total_loss += (batch_class).item()
            preds   = class_logits.argmax(dim=1)         
            correct += (preds == target_class).sum().item()
            total   += preds.size(0)
            tim = time.perf_counter()
            sys.stdout.write(
                f"\rBatch: {batch_idx + 1}/{len(val_dataloader)} | "
                f"Progress: {((batch_idx + 1) / len(val_dataloader) * 100):.1f}% | "
                f"Time: {(tim - start_time):.2f}s"
            )
            sys.stdout.flush()
    total_loss_val.append(total_loss)
    acc = (correct / total * 100) if total > 0 else 0.0
    val_acc_history.append(acc)
    print(
        f"\ntotal|{total_loss:.4f} | Val Acc: {acc:.2f}%"
    )
    print("=========================================================================")
    print("Validation Complete")
    print("=========================================================================")

val_acc_history=val_acc_history[3:]
def drawPlots():
    os.makedirs("plots", exist_ok=True)
    def to_float_list(lst):
        return [v.item() if hasattr(v, 'item') else float(v) for v in lst]
    n = len(total_loss_train)
    x = list(range(1, n + 1))
    plots = [
        (total_loss_train, total_loss_val, "Total Loss"),
    ]
    for train_data, val_data, title in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(x, to_float_list(train_data), label="Train", color="blue",   marker='o')
        plt.plot(x, to_float_list(val_data),   label="Val",   color="orange", marker='s')
        plt.title(f"{title} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        fname = title.lower().replace(" ", "_")
        plt.savefig(f"plots/{fname}.png", dpi=150)
        plt.close()
    if val_acc_history:
        plt.figure(figsize=(10, 6))
        plt.plot(x, to_float_list(val_acc_history), label="Val Acc (%)", color="green", marker='o')
        plt.axhline(y=65, color='red', linestyle='--', label="Target 65%")
        plt.title("Validation Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/val_accuracy.png", dpi=150)
        plt.close()
    print("All plots saved to plots/")

if __name__ == "__main__":
    train(isPreviouslyTrained=False, isNewData=True)