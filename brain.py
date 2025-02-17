import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np
from fbtree import Move

class ResBlock(nn.Module):
    """残差块，用于深层特征提取"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class PathEncoder(nn.Module):
    """路径编码器，将移动序列编码为固定长度的表示"""
    def __init__(self, board_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.board_size = board_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 位置编码
        self.move_embedding = nn.Embedding(
            board_size * board_size + 1,  # 增加一个特殊token用于空路径
            embedding_dim
        )
        
        # LSTM用于序列处理
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # 空路径的特殊token
        self.empty_token = board_size * board_size
        
    def forward(self, moves: List[List[Move]], device: str) -> torch.Tensor:
        # 处理空路径的情况
        if not moves or not moves[0]:
            batch_size = 1 if not moves else len(moves)
            # 使用特殊token生成空路径表示
            empty_input = torch.tensor([[self.empty_token]], device=device).expand(batch_size, 1)
            embedded = self.move_embedding(empty_input)
            lstm_out, _ = self.lstm(embedded)
            return lstm_out[:, -1]  # 返回最后一个状态
            
        # 正常处理非空路径
        max_len = max(len(path) for path in moves)
        batch_size = len(moves)
        
        # 创建填充的输入张量
        padded_moves = torch.zeros(batch_size, max_len, device=device).long()
        mask = torch.zeros(batch_size, max_len, device=device).bool()
        
        for i, path in enumerate(moves):
            if path:  # 确保路径非空
                move_values = [move.value for move in path]
                padded_moves[i, :len(move_values)] = torch.tensor(move_values, device=device)
                mask[i, :len(move_values)] = True
            else:
                # 对空路径使用特殊token
                padded_moves[i, 0] = self.empty_token
                mask[i, 0] = True
        
        # 嵌入移动
        embedded = self.move_embedding(padded_moves)
        
        # LSTM处理
        lstm_out, _ = self.lstm(embedded)
        
        # 注意力处理
        attended, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            key_padding_mask=~mask
        )
        attended = attended.transpose(0, 1)
        
        # 提取最后一个有效状态
        path_lengths = mask.sum(dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        final_states = attended[batch_indices, path_lengths - 1]
        
        return final_states

class Brain(nn.Module):
    def __init__(self, board_size: int, device: str = 'cpu', learning_rate: float = 0.001):
        super().__init__()
        self.board_size = board_size
        self.device = device
        
        # 路径编码器
        self.path_encoder = PathEncoder(board_size)
        
        # 主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            ResBlock(32)
        )
        
        # 路径特征投影
        self.path_projection = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, board_size * board_size)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state: torch.Tensor, path: List[List[Move]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码历史路径
        path_features = self.path_encoder(path, self.device)
        
        # 处理棋盘状态
        x = self.backbone(state)
        
        # 路径特征投影并广播
        path_features = self.path_projection(path_features)
        path_features = path_features.view(-1, 32, 1, 1).expand(-1, -1, self.board_size, self.board_size)
        
        # 特征融合
        x = x + path_features
        
        # 计算策略和价值
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        
        return policy, value
        
    def get_suggested_paths(self, state: torch.Tensor, current_path: List[Move],
                      valid_moves: List[Tuple[int, int]], 
                      temperature: float = 1.0, num_suggestions: int = 3) -> List[Tuple[Tuple[int, int], List[Move], float]]:
        """
        获取建议的后续路径及其胜率
        """
        if not valid_moves:
            return []  # 如果没有合法移动，返回空列表
            
        with torch.no_grad():
            paths = [[move for move in current_path]]
            policy, value = self.forward(state, paths)
            policy = policy.squeeze()
            
            # 只保留合法移动的概率
            valid_indices = [row * self.board_size + col for row, col in valid_moves]
            valid_probs = policy[valid_indices]
            
            # 温度处理
            if temperature != 1.0:
                valid_probs = torch.pow(valid_probs, 1.0/temperature)
                valid_probs = valid_probs / valid_probs.sum()
            
            # 选择概率最高的几个移动
            top_k = min(num_suggestions, len(valid_moves))
            top_values, top_indices = torch.topk(valid_probs, top_k)
            
            suggestions = []
            for prob, idx in zip(top_values, top_indices):
                move_idx = valid_indices[idx]
                row, col = move_idx // self.board_size, move_idx % self.board_size
                
                # 创建预测路径
                new_path = current_path.copy()
                new_path.append(Move(move_idx))
                
                suggestions.append(
                    ((row, col), new_path, prob.item() * value.item())
                )
            
            return suggestions
            
    def learn(self, states: torch.Tensor, moves: torch.Tensor, 
              rewards: torch.Tensor, paths: List[List[Move]]) -> float:
        self.optimizer.zero_grad()
        
        policy, value = self.forward(states, paths)
        
        # 策略损失
        policy_loss = nn.CrossEntropyLoss()(policy, moves)
        
        # 价值损失
        value_loss = nn.MSELoss()(value.squeeze(), rewards)
        
        # 总损失
        total_loss = policy_loss + 0.2 * value_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
    # 在 Brain 类末尾添加这两个方法
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'board_size': self.board_size,
            'device': self.device
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        assert self.board_size == checkpoint['board_size']