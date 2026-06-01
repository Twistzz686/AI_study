import torch
from torch import nn
from torchsummary import summary


# ==================== AlexNet 模型定义 ====================
# AlexNet 是2012年ImageNet竞赛的冠军模型，深度学习的里程碑
# 论文：《ImageNet Classification with Deep Convolutional Neural Networks》

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Args:
            num_classes: 分类的类别数，ImageNet默认1000类
        """
        super(AlexNet, self).__init__()

        # ==================== 特征提取部分 ====================
        # 卷积 + 池化层，提取图像特征
        self.features = nn.Sequential(
            # 第一层：卷积 + ReLU + 池化
            # 输入: 3×224×224 → 输出: 96×55×55
            # 参数: in_channels=3, out_channels=96, kernel=11, stride=4, padding=2
            nn.Conv2d(3, 96, 11, 4, 2),  # 原始论文是 stride=4，你这里是6，会改变输出尺寸
            nn.ReLU(inplace=True),  # inplace=True 节省内存

            # 最大池化：3×3，步长=2，输出: 96×27×27
            nn.MaxPool2d(3, 2),

            # 第二层：卷积 + ReLU + 池化
            # 输入: 96×27×27 → 输出: 256×27×27（padding=2保持尺寸）
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            # 池化后输出: 256×13×13
            nn.MaxPool2d(3, 2),

            # 第三层：卷积 + ReLU（无池化）
            # 输入: 256×13×13 → 输出: 384×13×13
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 第四层：卷积 + ReLU（无池化）
            # 输入: 384×13×13 → 输出: 384×13×13
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 第五层：卷积 + ReLU + 池化
            # 输入: 384×13×13 → 输出: 256×13×13
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            # 池化后输出: 256×6×6
            nn.MaxPool2d(3, 2),
        )

        # ==================== 分类器部分 ====================
        # 全连接层，将特征映射到类别概率
        self.classifier = nn.Sequential(
            # Dropout 防止过拟合，以0.5概率随机丢弃神经元
            nn.Dropout(0.5),
            # 第一层全连接：256×6×6 = 9216 → 4096
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            # 第二层全连接：4096 → 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 输出层：4096 → num_classes（如1000）
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 (batch, 3, 224, 224)
        Returns:
            输出张量，形状 (batch, num_classes)
        """
        # 特征提取
        x = self.features(x)  # (batch, 256, 6, 6)

        # 展平：将 (batch, 256, 6, 6) → (batch, 256*6*6)
        x = torch.flatten(x, 1)  # 从第1维开始展平（保留batch维度）

        # 全连接分类
        x = self.classifier(x)
        return x


# ==================== 程序入口 ====================
if __name__ == '__main__':
    # 将模型移到GPU（如果可用）
    model = AlexNet().to("cuda")

    # 打印模型摘要信息
    # 参数：模型实例，输入形状 (channels, height, width)
    summary(model, (3, 224, 224))