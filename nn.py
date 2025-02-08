import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """带残差连接的卷积块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = nn.Conv2d(channels, channels, 1)  # 空间注意力
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # 空间注意力机制
        att = torch.sigmoid(self.attention(x))
        x = x * att
        
        x += identity  # 残差连接
        return F.relu(x)

class PolicyHead(nn.Module):
    """策略头：输出落子概率分布"""
    def __init__(self, in_channels, board_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2*board_size**2, board_size**2 + 1)  # +1表示放弃手
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)

class ValueHead(nn.Module):
    """价值头：输出局面评估值"""
    def __init__(self, in_channels, board_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size**2, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class AlphaGomokuNet(nn.Module):
    """五子棋神经网络主体"""
    def __init__(self, board_size=15, history=8, channels=256, num_blocks=20):
        super().__init__()
        self.board_size = board_size
        
        # 输入层：当前局面 + 历史记录 + 特征平面
        self.input_conv = nn.Conv2d(history*2 + 3, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # 残差塔
        self.res_blocks = nn.Sequential(*[
            ConvBlock(channels) for _ in range(num_blocks)
        ])
        
        # 全局注意力模块
        self.attention_conv = nn.Conv2d(channels, channels, 1)
        self.attention_fc = nn.Linear(channels, board_size**2)
        
        # 输出头
        self.policy_head = PolicyHead(channels, board_size)
        self.value_head = ValueHead(channels, board_size)
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 输入处理
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # 残差计算
        x = self.res_blocks(x)
        
        # 全局注意力增强
        batch_size = x.size(0)
        att = F.avg_pool2d(x, self.board_size).view(batch_size, -1)
        att = torch.sigmoid(self.attention_fc(att))
        att = att.view(batch_size, -1, self.board_size, self.board_size)
        x = x * att
        
        # 双头输出
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

