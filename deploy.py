from farch2 import Model, load
import os
import cv2
import torch
import pandas as pd
import torchvision.transforms.functional as tf

def makeVideoTensor(file: str=r"C:\Users\CSIS workstation\OneDrive\Desktop\edo\data\CrackDataVal\td-1\psample-13_1.csv"):
    df = pd.read_csv(file)
    imgs = []
    for _, row in df.iterrows():
        imgAdd = row["img"]
        frame = cv2.imread(imgAdd)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameTensor = torch.from_numpy(frame).float() / 255.0
        frameTensor = frameTensor.permute(2, 0, 1)
        frameTensor = tf.resize(frameTensor, [640, 640], interpolation=tf.InterpolationMode.BICUBIC)
        imgs.append(frameTensor)
    videoTensor = torch.stack(imgs, dim=0).unsqueeze(0)
    return videoTensor

def makeVideoFromImg(path:str=r"test\1.webp"):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tensor=torch.from_numpy(img).float()/255.0
    tensor=tensor.permute(2,0,1)
    tensor=tf.resize(tensor,[640,640],interpolation=tf.InterpolationMode.BICUBIC)
    vidT=[tensor for _ in range(8)]
    vidT=torch.stack(vidT,dim=0)
    vidT=vidT.unsqueeze(dim=0)
    return vidT

def makeVideo(file:str=r"test\Recording 2026-06-19 101954.mp4"):
    cap=cv2.VideoCapture(file)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    indices=range(90,150,8)
    indices=list(indices)
    count=0
    frameList=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        if count in indices:
            frameList.append(frame)
        if count> indices[-1]:
            break
        count+=1
    tensor=[]
    for _ in frameList:
        img=cv2.cvtColor(_,cv2.COLOR_BGR2RGB)
        img=torch.from_numpy(img).float()/255.0
        img=img.permute(2,0,1)
        img=tf.resize(img,[640,640],interpolation=tf.InterpolationMode.BICUBIC)
        tensor.append(img)
    vidT=torch.stack(tensor,dim=0)
    vidT=vidT.unsqueeze(0)
    return vidT

if __name__ == "__main__":
    model = Model()
    load(model,r"results\finalModel.pth")
    model.eval()
    vidT = makeVideoFromImg()
    with torch.no_grad():
        class_logits = model(vidT)
        probs = torch.softmax(class_logits, dim=1)
        print(probs)
