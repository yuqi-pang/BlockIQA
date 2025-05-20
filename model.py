import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        resnet = resnet50(weights='DEFAULT')
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) for _ in range(4)
        ])

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        p4 = self.fpn_layers[0](x4)
        p3 = self._upsample_add(self.fpn_layers[1](x3), p4)
        p2 = self._upsample_add(self.fpn_layers[2](x2), p3)
        p1 = self._upsample_add(self.fpn_layers[3](x1), p2)

        p4 = self.smooth_layers[0](p4)
        p3 = self.smooth_layers[1](p3)
        p2 = self.smooth_layers[2](p2)
        p1 = self.smooth_layers[3](p1)

        out = torch.cat([p1, p2, p3, p4], dim=1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fpn = FPN()
        self.ssl_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.final_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        fpn_features = self.fpn(x)
        ssl_features = self.ssl_model(x)
        fused_features = torch.cat((fpn_features, ssl_features), dim=1)
        fused_features = self.fusion_layer(fused_features)
        quality_score = self.final_layer(fused_features)
        return quality_score, fpn_features

class ImageQualityAggregator(nn.Module):
    def __init__(self, n_blocks):
        super(ImageQualityAggregator, self).__init__()
        self.n_blocks = n_blocks
        self.aggregator = nn.Sequential(
            nn.Linear(1024 + 1, 128),  # 增加质量得分维度
            nn.BatchNorm1d(128),  # 添加Batch Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # 添加Batch Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout
            nn.Linear(64, 1),
            nn.Sigmoid()  # 使用Sigmoid将输出限制在0到1之间
        )

    def gaussian_similarity(self, features):
        # 计算高斯相似度矩阵
        similarity_matrix = torch.zeros((self.n_blocks ** 2, self.n_blocks ** 2))
        feature_dim = features.shape[1]  # 特征向量的维度
        for i in range(self.n_blocks ** 2):
            for j in range(self.n_blocks ** 2):
                if i != j:
                    # 计算余弦相似度
                    cos_sim = F.cosine_similarity(features[i], features[j], dim=0, eps=1e-6)
                    # 动态调整高斯函数的标准差
                    sigma = torch.sqrt(feature_dim * (0.5 + 0.5 * cos_sim))  # 使用特征向量维度的均值
                    diff = features[i] - features[j]
                    similarity_matrix[i, j] = torch.exp(-torch.norm(diff) ** 2 / (2 * sigma ** 2))
                else:
                    # 对角线元素设置为1，因为向量与自身的相似度为1
                    similarity_matrix[i, j] = 1
        return similarity_matrix

    def forward(self, features, scores):
        similarity_matrix = self.gaussian_similarity(features)
        combined_features = torch.cat((features, scores.unsqueeze(1)), dim=1)  # 合并特征和质量得分
        aggregated_score = self.aggregator(combined_features)
        # 使用相似度矩阵来调整最终得分
        adjusted_score = (aggregated_score * similarity_matrix.mean(dim=1).to(device)).mean()
        return adjusted_score

class CombinedModel(nn.Module):
    def __init__(self, basemodel, image_quality_aggregator):
        super(CombinedModel, self).__init__()
        self.basemodel = basemodel
        self.image_quality_aggregator = image_quality_aggregator

    def forward(self, x):
        pred_scores, block_features = self.basemodel(x)
        pred_scores = pred_scores.squeeze()  # 去掉多余的维度
        final_score = self.image_quality_aggregator(block_features, pred_scores)
        return final_score