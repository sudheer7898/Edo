import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregatorToTransformer(nn.Module):
    def __init__(self, in_channels_list, d_model, target_size=(80, 80)):
        super().__init__()
        self.target_size = target_size
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, d_model, kernel_size=1) for in_ch in in_channels_list
        ])
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list), dtype=torch.float32))
        self.smooth = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.epsilon = 1e-4
    def forward(self, features):
        fused_features = 0
        positive_weights = F.relu(self.fusion_weights)
        weight_sum = torch.sum(positive_weights) + self.epsilon
        for i, x in enumerate(features):
            x_proj = self.projections[i](x)
            if x_proj.shape[-2:] != self.target_size:
                x_rescaled = F.interpolate(x_proj, size=self.target_size, mode='bilinear', align_corners=False)
            else:
                x_rescaled = x_proj
            normalized_weight = positive_weights[i] / weight_sum
            fused_features += normalized_weight * x_rescaled
        return self.smooth(fused_features).permute(0, 2, 3, 1)


if __name__ == "__main__":
    BT = 16
    P3 = torch.rand(BT, 64, 40, 40)
    P4 = torch.rand(BT, 128, 20, 20)
    P5 = torch.rand(BT, 256, 10, 10)
    aggregator = FeatureAggregatorToTransformer(in_channels_list=[64, 128, 256], d_model=256, target_size=(20, 20))
    transformer_input = aggregator([P3, P4, P5])
    print("Shape ready for spatial reshaping:", transformer_input.shape) 