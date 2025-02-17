import torch
from typing import List, Tuple, Optional, Dict
from collections import deque
import random
import uuid
import math
from board import Board
from brain import Brain
from fbtree import FiberTree, Move

def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Mask:
    """带生命周期的 AI 智能体"""
    
    def __init__(self, board_size: int = 15, device: str = 'cpu', 
                 generation: int = 0, parent_id: str = None):
        # 基础属性
        self.board_size = board_size
        self.device = device
        self.brain = Brain(board_size, device)
        self.memory = FiberTree()
        self.experiences = deque(maxlen=10000)
        self.current_game_states = []
        
        # 生命周期属性
        self.id = str(uuid.uuid4())[:8]  # 唯一标识符
        self.generation = generation      # 代数
        self.parent_id = parent_id       # 父代ID
        self.age = 0                     # 年龄（夜昼数）
        self.health = 100                # 生命值
        self.digestion_count = 0         # 当前周期消化次数
        self.battle_count = 0            # 当前周期对战次数
        self.total_games = 0             # 总对局数
        self.win_count = 0               # 总胜利数
        self.performance = 0.0           # 性能评分
        
    def start_game(self):
        """开始新游戏"""
        self.memory.start_adding_mode()
        self.current_game_states = []
        
    def end_game(self, won: bool):
        """结束游戏并更新统计"""
        self.memory.end_adding_mode()
        self.memory.update_statistics(won)
        
        if won is not None:  # won可能是True/False，不包括平局
            final_path = self.memory.get_complete_path()
            path_length = len(final_path)
            
            for idx, (board_state, move, player) in enumerate(self.current_game_states):
                position_weight = idx / path_length
                if (player == 'X' and won) or (player == 'O' and not won):
                    reward = 0.5 + 0.5 * position_weight
                else:
                    reward = 0.5 - 0.5 * position_weight
                    
                self.add_experience(board_state, move, player, reward, final_path[:idx+1])
            
            self.learn(batch_size=min(64, len(self.current_game_states)))
        
        self.current_game_states = []
        
    def digest(self) -> bool:
        """自我对弈（消化）
        Returns:
            bool: 是否存活
        """
        if self.health <= 0:
            return False
            
        # 自我对弈一局
        self.start_game()
        board = Board(self.board_size, 5)  # 假设都是五子棋规则
        moves_count = 0
        
        while not board.is_full() and not board.has_winner():
            valid_moves = board.get_valid_moves()
            current_player = 'X' if moves_count % 2 == 0 else 'O'
            board_state = board.get_board()
            
            move = self.get_move(board_state, valid_moves, current_player)
            if move and board.make_move(move[0], move[1], current_player):
                moves_count += 1
                if board.check_win(move[0], move[1], current_player):
                    break
        
        # 更新状态
        self.digestion_count += 1
        self.total_games += 1
        
        # 根据年龄计算消化消耗
        consumption = 5 + min(15, self.age // 10)  # 年龄越大，消耗越大
        self.health = max(0, self.health - consumption)
        
        return self.health > 0
        
    def battle(self, opponent: 'Mask', game_id: str = None) -> int:
        """与其他 Mask 对战
        Args:
            opponent: 对手 Mask
            game_id: 对局 ID，用于记录
            
        Returns:
            int: 1 表示胜利，0 表示平局，-1 表示失败
        """
        if self.health <= 0 or opponent.health <= 0:
            return 0
            
        board = Board(self.board_size, 5)
        self.start_game()
        opponent.start_game()
        moves_count = 0
        
        while not board.is_full() and not board.has_winner():
            current_player = 'X' if moves_count % 2 == 0 else 'O'
            current_mask = self if current_player == 'X' else opponent
            valid_moves = board.get_valid_moves()
            board_state = board.get_board()
            
            move = current_mask.get_move(board_state, valid_moves, current_player)
            if move and board.make_move(move[0], move[1], current_player):
                moves_count += 1
                if board.check_win(move[0], move[1], current_player):
                    break
        
        # 计算结果
        result = 0
        if board.has_winner():
            winner = board.get_winner()
            result = 1 if winner == 'X' else -1
            self.end_game(result > 0)
            opponent.end_game(result < 0)
        else:
            self.end_game(None)
            opponent.end_game(None)
        
        # 更新状态
        self._update_battle_status(result)
        opponent._update_battle_status(-result)
        
        return result
        
    def _update_battle_status(self, result: int):
        """更新对战后的状态
        Args:
            result: 1 胜利，0 平局，-1 失败
        """
        self.battle_count += 1
        self.total_games += 1
        
        # 计算基础奖惩
        base_reward = 20  # 基础奖励/惩罚值
        age_factor = max(0.2, 1 - self.age / 50)  # 年龄影响因子
        
        if result > 0:
            self.win_count += 1
            reward = base_reward * age_factor
            self.health = min(100, self.health + reward)
        elif result < 0:
            penalty = base_reward * (2 - age_factor)  # 年龄大的惩罚更重
            self.health = max(0, self.health - penalty)
        else:  # 平局
            penalty = base_reward * 0.5
            self.health = max(0, self.health - penalty)
        
        # 更新性能评分
        self.performance = self.win_count / max(1, self.total_games)
        
    def get_move(self, board: List[List[str]], valid_moves: List[Tuple[int, int]], 
                 player: str, temperature: float = 1.0) -> Optional[Tuple[int, int]]:
        """决定下一步移动"""
        if not valid_moves:
            return None
            
        current_path = self.memory.get_complete_path()
        state = self._board_to_tensor(board, player)
        
        suggested_paths = self.brain.get_suggested_paths(
            state, 
            current_path,
            valid_moves, 
            temperature
        )
        
        if not suggested_paths:
            return None
            
        best_move = suggested_paths[0][0]  # 取建议路径中的第一个移动
        
        # 记录状态
        self.current_game_states.append((board, best_move, player))
        
        # 记录到历史记忆
        self.memory.add_move(Move(self._encode_move(best_move[0], best_move[1])))
        
        return best_move
        
    def reproduce(self) -> Optional['Mask']:
        """繁殖后代
        Returns:
            Optional[Mask]: 如果条件满足则返回新的 Mask，否则返回 None
        """
        if self.health < 50:  # 需要足够的生命值
            return None
            
        # 创建子代
        child = Mask(
            board_size=self.board_size,
            device=self.device,
            generation=self.generation + 1,
            parent_id=self.id
        )
        
        # 继承父代的大脑参数（带有小概率突变）
        child.brain.load_state_dict(self.brain.state_dict())
        
        # 随机突变
        if random.random() < 0.1:  # 10% 的突变概率
            with torch.no_grad():
                for param in child.brain.parameters():
                    if random.random() < 0.1:  # 每个参数 10% 的突变概率
                        mutation = torch.randn_like(param) * 0.1
                        param.add_(mutation)
        
        # 消耗一定生命值
        self.health = max(0, self.health - 30)
        
        return child
        
    def update_age(self):
        """更新年龄，重置周期计数"""
        self.age += 1
        self.digestion_count = 0
        self.battle_count = 0
        
        # 自然衰老
        age_decay = self.age // 20  # 每20个周期增加1点衰老
        self.health = max(0, self.health - age_decay)
        
    def _encode_move(self, row: int, col: int) -> int:
        """将移动坐标编码为整数"""
        return row * self.board_size + col
        
    def _decode_move(self, value: int) -> Tuple[int, int]:
        """将整数解码为移动坐标"""
        return value // self.board_size, value % self.board_size
        
    def _board_to_tensor(self, board: List[List[str]], player: str) -> torch.Tensor:
        """将棋盘状态转换为神经网络输入格式"""
        tensor = torch.zeros(1, 3, self.board_size, self.board_size, device=self.device)
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    tensor[0, 0, i, j] = 1.0  # 己方棋子
                elif board[i][j] != ' ':
                    tensor[0, 1, i, j] = 1.0  # 对方棋子
                else:
                    tensor[0, 2, i, j] = 1.0  # 空位
                    
        return tensor
        
    def add_experience(self, board: List[List[str]], move: Tuple[int, int], 
                      player: str, reward: float, path: List[Move]):
        """记录训练经验"""
        self.experiences.append({
            'state': self._board_to_tensor(board, player),
            'move': self._encode_move(move[0], move[1]),
            'reward': reward,
            'path': path
        })
        
    def learn(self, batch_size: int = 32) -> Optional[float]:
        """从经验中学习"""
        if len(self.experiences) < batch_size:
            return None
            
        indices = torch.randperm(len(self.experiences))[:batch_size]
        batch = [self.experiences[i] for i in indices]
        
        states = torch.cat([exp['state'] for exp in batch])
        moves = torch.tensor([exp['move'] for exp in batch], device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], device=self.device).float()
        paths = [exp['path'] for exp in batch]
        
        return self.brain.learn(states, moves, rewards, paths)
        
    def save(self, path: str):
        """保存模型和状态"""
        state = {
            'generation': self.generation,
            'age': self.age,
            'health': self.health,
            'total_games': self.total_games,
            'win_count': self.win_count,
            'performance': self.performance,
            'model_state': self.brain.state_dict(),
            'board_size': self.board_size,
            'id': self.id,
            'parent_id': self.parent_id
        }
        torch.save(state, path)
        
    def load(self, path: str):
        """加载模型和状态"""
        state = torch.load(path, map_location=self.device)
        self.generation = state['generation']
        self.age = state['age']
        self.health = state['health']
        self.total_games = state['total_games']
        self.win_count = state['win_count']
        self.performance = state['performance']
        self.board_size = state['board_size']
        self.id = state['id']
        self.parent_id = state['parent_id']
        self.brain.load_state_dict(state['model_state'])
        
    def get_state(self) -> dict:
        """获取当前状态信息"""
        return {
            'id': self.id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'age': self.age,
            'health': self.health,
            'total_games': self.total_games,
            'win_count': self.win_count,
            'performance': self.performance,
            'board_size': self.board_size
        }
        
    def is_alive(self) -> bool:
        """检查是否存活"""
        return self.health > 0