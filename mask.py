import torch
from torch import nn
import numpy as np
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
        """决定下一步移动，使用加权随机选择策略"""
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
        
        # 智能路径选择逻辑
        if temperature < 0.5 or random.random() < 0.7:  # 探索与利用的平衡
            # 利用模式：根据胜率加权选择
            moves, paths, probs = zip(*suggested_paths)
            probs = torch.tensor(probs)
            probs = torch.softmax(probs / max(0.1, temperature), dim=0)  # 温度缩放
            chosen_idx = torch.multinomial(probs, 1).item()
            best_move = moves[chosen_idx]
            selected_path = paths[chosen_idx]
        else:
            # 探索模式：在前N个建议中随机选择
            explore_range = min(len(suggested_paths), 3)  # 探索范围
            chosen_idx = random.randrange(explore_range)
            best_move, selected_path, _ = suggested_paths[chosen_idx]
        
        # 记录状态
        self.current_game_states.append((board, best_move, player))
        
        # 记录到历史记忆
        self.memory.add_move(Move(self._encode_move(best_move[0], best_move[1])))
        
        return best_move
        
    def reproduce(self) -> Optional['Mask']:
        """高级繁殖方法，具有动态突变率和参数特异性突变
        
        Returns:
            Optional[Mask]: 如果条件满足则返回新的 Mask，否则返回 None
        """
        # 健康检查 - 需要足够的生命值才能繁殖
        if self.health < 50 or not self.is_alive():
            return None
            
        # 创建子代
        child = Mask(
            board_size=self.board_size,
            device=self.device,
            generation=self.generation + 1,
            parent_id=self.id
        )
        
        # 继承父代的大脑参数
        child.brain.load_state_dict(self.brain.state_dict())
        
        # 动态突变率 - 根据父代性能和年龄调整
        base_mutation_prob = 0.1  # 基础突变概率
        
        # 性能因子 - 性能越好，突变率越低
        performance_factor = max(0.5, 1.0 - self.performance)
        
        # 年龄因子 - 年龄越大，突变率略微增加（促进探索）
        age_factor = min(1.5, 1.0 + (self.age / 50) * 0.5)
        
        # 健康因子 - 健康越好，突变率越低（稳定遗传）
        health_factor = max(0.8, 1.0 - (self.health / 100) * 0.2)
        
        # 计算最终突变率
        mutation_prob = base_mutation_prob * performance_factor * age_factor * health_factor
        
        # 参数特异性突变
        with torch.no_grad():
            # 对不同层使用不同突变强度
            for name, param in child.brain.named_parameters():
                # 确定该参数的突变概率
                param_mutation_prob = mutation_prob
                
                # 策略网络和价值网络参数差异化突变
                if 'policy_head' in name:
                    param_mutation_prob *= 0.8  # 策略头参数更保守
                elif 'value_head' in name:
                    param_mutation_prob *= 1.2  # 价值头参数更激进
                    
                # 对卷积层和全连接层差异化处理
                if 'weight' in name and param.dim() > 1:
                    if param.dim() == 4:  # 卷积层
                        # 对卷积核逐个突变
                        for i in range(param.size(0)):
                            if random.random() < param_mutation_prob:
                                # 卷积核突变 - 保持结构特性
                                mutation_scale = 0.05 * random.random()
                                mutation = torch.randn_like(param[i]) * mutation_scale
                                param[i].add_(mutation)
                    else:  # 全连接层
                        if random.random() < param_mutation_prob:
                            # 全连接层突变
                            mutation_scale = 0.1 * random.random()
                            mutation = torch.randn_like(param) * mutation_scale
                            param.add_(mutation)
                elif 'bias' in name:
                    # 偏置项突变概率较低
                    if random.random() < param_mutation_prob * 0.5:
                        mutation_scale = 0.02 * random.random()
                        mutation = torch.randn_like(param) * mutation_scale
                        param.add_(mutation)
                else:
                    # 其他参数通用突变
                    if random.random() < param_mutation_prob:
                        mutation_scale = 0.08 * random.random()
                        mutation = torch.randn_like(param) * mutation_scale
                        param.add_(mutation)
        
        # 继承部分记忆（选择性记忆继承）
        if hasattr(self.memory, 'root') and self.memory.root:
            # 选择性复制重要的记忆路径
            important_fibers = self._select_important_fibers()
            for fiber in important_fibers:
                child.memory.start_adding_mode()
                for move in fiber:
                    child.memory.add_move(move)
                child.memory.end_adding_mode()
        
        # 消耗生命值，根据突变程度调整消耗
        mutation_cost = 30 * (1 + mutation_prob / base_mutation_prob * 0.5)
        self.health = max(0, self.health - mutation_cost)
        
        return child

    def _select_important_fibers(self) -> List[List[Move]]:
        """
        选择重要记忆路径传递给后代
        
        策略：
        1. 选择高胜率的路径
        2. 选择高访问次数的路径
        3. 保持路径多样性
        4. 限制总路径数量以避免过拟合
        
        Returns:
            List[List[Move]]: 重要记忆路径列表
        """
        important_fibers = []
        fibers_info = []
        max_fibers_to_transfer = 10  # 最大传递路径数
        
        # 只有当内存中有记忆时才进行处理
        if not hasattr(self.memory, 'root') or not self.memory.root or not self.memory.root.next_fibers:
            return important_fibers
        
        # 递归收集Fiber信息
        def collect_fibers(fiber, path_so_far=None):
            if path_so_far is None:
                path_so_far = []
            
            if not fiber.next_fibers:  # 叶节点
                if fiber.visit_count > 0:
                    complete_path = path_so_far + fiber.moves
                    win_rate = fiber.win_count / max(1, fiber.visit_count)
                    fibers_info.append({
                        'path': complete_path,
                        'visit_count': fiber.visit_count,
                        'win_rate': win_rate,
                        'depth': len(complete_path),
                        'confidence': win_rate * math.log(fiber.visit_count + 1)
                    })
                return
            
            # 继续递归遍历
            for next_fiber in fiber.next_fibers:
                collect_fibers(next_fiber, path_so_far + fiber.moves)
        
        # 从根节点开始收集
        collect_fibers(self.memory.root)
        
        # 如果没有收集到任何路径
        if not fibers_info:
            return important_fibers
        
        # 按不同指标排序并选择
        # 1. 高置信度路径（结合胜率和访问次数）
        confidence_sorted = sorted(fibers_info, key=lambda x: x['confidence'], reverse=True)
        high_confidence_paths = [item['path'] for item in confidence_sorted[:min(5, len(confidence_sorted))]]
        
        # 2. 高胜率但样本量适中的路径
        win_rate_sorted = sorted(
            [f for f in fibers_info if f['visit_count'] >= 3],  # 至少3次访问以确保统计稳定性
            key=lambda x: x['win_rate'], 
            reverse=True
        )
        high_winrate_paths = [item['path'] for item in win_rate_sorted[:min(3, len(win_rate_sorted))]]
        
        # 3. 选择一些长路径（可能代表完整对局）
        depth_sorted = sorted(fibers_info, key=lambda x: x['depth'], reverse=True)
        deep_paths = [item['path'] for item in depth_sorted[:min(2, len(depth_sorted))]]
        
        # 4. 选择一些较短但高频路径（开局策略）
        opening_candidates = [
            f for f in fibers_info 
            if 2 <= f['depth'] <= 6 and f['visit_count'] >= 5
        ]
        opening_sorted = sorted(opening_candidates, key=lambda x: x['visit_count'], reverse=True)
        opening_paths = [item['path'] for item in opening_sorted[:min(3, len(opening_sorted))]]
        
        # 合并所有选中路径，确保不重复
        all_paths = []
        all_paths.extend(high_confidence_paths)
        all_paths.extend(high_winrate_paths)
        all_paths.extend(deep_paths)
        all_paths.extend(opening_paths)
        
        # 去重 - 基于路径的字符串表示
        unique_paths = {}
        for path in all_paths:
            path_str = ','.join([str(move.value) for move in path])
            if path_str not in unique_paths:
                unique_paths[path_str] = path
        
        # 构建最终的路径列表，控制数量
        important_fibers = list(unique_paths.values())[:max_fibers_to_transfer]
        
        # 随机保留一些路径以增加多样性（如果路径足够多）
        if len(important_fibers) > max_fibers_to_transfer // 2:
            # 保留高价值的一半，剩下的随机选择
            guaranteed = important_fibers[:max_fibers_to_transfer // 2]
            candidates = important_fibers[max_fibers_to_transfer // 2:]
            random_selected = random.sample(
                candidates, 
                min(len(candidates), max_fibers_to_transfer - len(guaranteed))
            )
            important_fibers = guaranteed + random_selected
        
        return important_fibers
        
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
        
    def learn(self, batch_size: int = 32, learning_iterations: int = 3) -> Dict[str, float]:
        """从经验中学习，使用先进的学习策略
        
        Args:
            batch_size: 单次学习的样本数量
            learning_iterations: 学习迭代次数，增加可加深学习
            
        Returns:
            Dict[str, float]: 学习统计信息，包括各类损失值和学习指标
        """
        if len(self.experiences) < batch_size:
            return {'status': 'insufficient_data', 'loss': 0.0}
        
        stats = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'consistency_loss': 0.0,
            'iterations': 0
        }
        
        # 计算经验重要性 - 基于稀有度和时间性
        experience_weights = self._calculate_experience_weights()
        
        for iteration in range(learning_iterations):
            # 优先经验采样（有偏采样）
            if random.random() < 0.7:  # 70%概率使用加权采样
                indices = self._weighted_sample(experience_weights, batch_size)
            else:  # 30%概率完全随机采样以增加多样性
                indices = torch.randperm(len(self.experiences))[:batch_size]
                
            batch = [self.experiences[i] for i in indices]
            
            # 准备数据
            states = torch.cat([exp['state'] for exp in batch])
            moves = torch.tensor([exp['move'] for exp in batch], device=self.device)
            rewards = torch.tensor([exp['reward'] for exp in batch], device=self.device).float()
            paths = [exp['path'] for exp in batch]
            
            # 是否包含旧预测值（如果有）
            has_old_predictions = all('old_policy' in exp for exp in batch)
            if has_old_predictions:
                old_policies = torch.cat([exp['old_policy'] for exp in batch])
                old_values = torch.cat([exp['old_value'] for exp in batch])
            
            # 获取当前预测
            policies, values = self.brain.forward(states, paths)
            
            # 保存当前预测用于后续PPO风格更新
            for i, idx in enumerate(indices):
                if idx < len(self.experiences):  # 防止索引错误
                    self.experiences[idx]['old_policy'] = policies[i:i+1].detach()
                    self.experiences[idx]['old_value'] = values[i:i+1].detach()
            
            # 计算策略损失（带熵正则化）
            policy_loss = nn.CrossEntropyLoss(reduction='none')(policies, moves)
            policy_probs = torch.softmax(policies, dim=1)
            entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=1)
            
            # 价值损失（Huber损失更稳定）
            value_loss = nn.SmoothL1Loss(reduction='none')(values.squeeze(), rewards)
            
            # PPO风格的策略约束（如果有旧预测）
            if has_old_predictions:
                old_policy_probs = torch.softmax(old_policies, dim=1)
                # 提取每个动作的概率
                move_indices = moves.unsqueeze(1)
                new_probs = torch.gather(policy_probs, 1, move_indices).squeeze()
                old_probs = torch.gather(old_policy_probs, 1, move_indices).squeeze()
                
                # 计算比率并裁剪
                ratios = new_probs / (old_probs + 1e-10)
                clipped_ratios = torch.clamp(ratios, 0.8, 1.2)
                ppo_loss = -torch.min(
                    ratios * (rewards - old_values.squeeze()),
                    clipped_ratios * (rewards - old_values.squeeze())
                )
                
                # 自我模仿学习损失（对高回报动作的模仿）
                high_value_mask = rewards > rewards.mean()
                imitation_loss = torch.zeros_like(policy_loss)
                if high_value_mask.sum() > 0:
                    kl_div = nn.KLDivLoss(reduction='none')(
                        torch.log_softmax(policies[high_value_mask], dim=1),
                        policy_probs[high_value_mask]
                    ).sum(dim=1)
                    imitation_loss[high_value_mask] = kl_div
                    
                combined_policy_loss = policy_loss + 0.5 * ppo_loss + 0.3 * imitation_loss
            else:
                combined_policy_loss = policy_loss
            
            # 经验权重加权的总损失
            batch_weights = torch.tensor([experience_weights[i] for i in indices], 
                                        device=self.device).float()
            weighted_policy_loss = (combined_policy_loss * batch_weights).mean()
            weighted_value_loss = (value_loss * batch_weights).mean()
            
            # 添加正则化损失
            l2_reg = 0.0
            for param in self.brain.parameters():
                l2_reg += 0.0001 * torch.sum(param ** 2)
                
            # 总损失
            entropy_bonus = 0.01 * entropy.mean()  # 鼓励探索
            consistency_loss = 0.0
            
            # 如果批次中有相似状态，鼓励一致性
            if states.size(0) > 1:
                similarity_matrix = self._compute_state_similarity(states)
                similar_pairs = (similarity_matrix > 0.85).float() - torch.eye(states.size(0), device=self.device)
                if similar_pairs.sum() > 0:
                    policy_diffs = torch.cdist(policy_probs, policy_probs, p=1)
                    consistency_loss = (similar_pairs * policy_diffs).sum() / (similar_pairs.sum() + 1e-6)
            
            total_loss = (
                weighted_policy_loss + 
                0.5 * weighted_value_loss - 
                entropy_bonus + 
                l2_reg +
                0.2 * consistency_loss
            )
            
            # 优化器更新
            self.brain.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
            self.brain.optimizer.step()
            
            # 更新统计
            stats['total_loss'] += total_loss.item()
            stats['policy_loss'] += weighted_policy_loss.item()
            stats['value_loss'] += weighted_value_loss.item()
            stats['entropy'] += entropy.mean().item()
            stats['consistency_loss'] += consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss
            stats['iterations'] += 1
            
            # 早停 - 如果损失过低则提前结束
            if total_loss.item() < 0.01:
                break
                
        # 计算平均损失
        for key in ['total_loss', 'policy_loss', 'value_loss', 'entropy', 'consistency_loss']:
            if stats['iterations'] > 0:
                stats[key] /= stats['iterations']
        
        # 定期回顾性经验压缩（长期记忆管理）
        if random.random() < 0.05:  # 5%概率执行
            self._consolidate_experiences()
            
        return stats

    def _weighted_sample(self, weights: List[float], sample_size: int) -> List[int]:
        """加权采样，确保重要经验更频繁地被学习
        
        Args:
            weights: 每个经验的权重列表
            sample_size: 要采样的经验数量
            
        Returns:
            List[int]: 采样得到的索引列表
        """
        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = torch.clamp(weights_tensor, min=0.01)  # 确保每个样本有最小概率被选中
        
        # 温度采样 - 使概率分布更平滑或更尖锐
        temperature = max(0.5, 1.0 - 0.05 * self.age)  # 随年龄增长降低温度
        weights_tensor = weights_tensor ** (1.0 / temperature)
        
        # 正则化为概率
        probs = weights_tensor / weights_tensor.sum()
        
        # 多项分布采样
        indices = torch.multinomial(probs, 
                                sample_size, 
                                replacement=len(weights) < sample_size)
                                
        return indices.tolist()

    def _calculate_experience_weights(self) -> List[float]:
        """计算每个经验样本的重要性权重
        
        考虑因素:
        1. 回报值 - 高回报经验更有价值
        2. 稀有度 - 不常见的状态更有价值
        3. 时间衰减 - 新的经验相对更重要
        4. TD误差 - 预测误差大的样本更需要学习
        
        Returns:
            List[float]: 权重列表
        """
        if not self.experiences:
            return []
            
        num_exp = len(self.experiences)
        weights = np.ones(num_exp)
        
        # 1. 回报值重要性
        rewards = np.array([exp['reward'] for exp in self.experiences])
        reward_weights = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-6)
        weights *= (1.0 + reward_weights)
        
        # 2. 稀有度评分（基于状态哈希的计数）
        state_hashes = {}
        for exp in self.experiences:
            # 简化的状态哈希
            state_flat = exp['state'].cpu().numpy().flatten()
            state_hash = hash(state_flat.tobytes()[:100])  # 取前100个字节防止计算量过大
            state_hashes[state_hash] = state_hashes.get(state_hash, 0) + 1
        
        rarity_weights = np.array([1.0 / state_hashes.get(hash(exp['state'].cpu().numpy().flatten().tobytes()[:100]), 1) 
                                for exp in self.experiences])
        rarity_weights = rarity_weights / (rarity_weights.max() + 1e-6)
        weights *= (1.0 + 0.5 * rarity_weights)  # 稀有样本获得50%额外权重
        
        # 3. 时间衰减 - 指数衰减，最近的经验更重要
        recency_factor = 0.97  # 每个位置衰减3%
        position_weights = np.array([recency_factor ** (num_exp - i - 1) for i in range(num_exp)])
        weights *= position_weights
        
        # 4. TD误差（如果有预测值）
        if all('old_value' in exp for exp in self.experiences):
            td_errors = np.array([
                abs(exp['reward'] - exp['old_value'].item())
                if 'old_value' in exp else 1.0
                for exp in self.experiences
            ])
            if td_errors.max() > td_errors.min():
                td_weights = (td_errors - td_errors.min()) / (td_errors.max() - td_errors.min())
                weights *= (1.0 + 0.7 * td_weights)  # TD误差大的获得70%额外权重
        
        # 归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()
            
        return weights.tolist()

    def _compute_state_similarity(self, states: torch.Tensor) -> torch.Tensor:
        """计算批次内状态之间的相似度
        
        Args:
            states: 状态张量 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 相似度矩阵 [batch_size, batch_size]
        """
        # 将状态展平为向量
        flat_states = states.view(states.size(0), -1)
        
        # 计算余弦相似度
        norm_states = flat_states / (torch.norm(flat_states, dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(norm_states, norm_states.transpose(0, 1))
        
        return similarity

    def _consolidate_experiences(self):
        """整合和压缩经验记忆
        
        策略:
        1. 删除低价值的冗余经验
        2. 保留代表性经验样本
        3. 确保记忆的多样性
        """
        if len(self.experiences) < 1000:
            return  # 仅在经验足够多时执行
            
        # 计算经验重要性
        weights = self._calculate_experience_weights()
        
        # 识别低价值经验（保留80%的经验）
        threshold = sorted(weights)[int(len(weights) * 0.2)]
        
        # 聚类保留代表性样本
        # 简化版：优先保留高权重和最近的经验
        new_experiences = []
        
        # 1. 首先保留高价值经验（权重>阈值）
        for i, (exp, w) in enumerate(zip(self.experiences, weights)):
            if w > threshold:
                new_experiences.append(exp)
                
        # 2. 从剩余经验中抽样保留一定数量的经验
        remaining = [exp for i, exp in enumerate(self.experiences) 
                    if weights[i] <= threshold]
        
        # 保留最近的30%经验
        recent_count = int(len(self.experiences) * 0.3)
        new_experiences.extend(self.experiences[-recent_count:])
        
        # 随机保留一些剩余经验，提高多样性
        remaining_needed = self.experiences.maxlen - len(new_experiences)
        if remaining_needed > 0 and remaining:
            random_sample = random.sample(remaining, 
                                        min(remaining_needed, len(remaining)))
            new_experiences.extend(random_sample)
        
        # 更新经验池
        self.experiences = deque(new_experiences, maxlen=self.experiences.maxlen)
        
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