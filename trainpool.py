import random
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import logging
from tqdm import tqdm
from mask import Mask, set_seed

class TrainPool:
    def __init__(self, 
                 board_size: int = 15,
                 initial_size: int = 20,
                 min_pool_size: int = 10,
                 max_pool_size: int = 30,
                 device: str = 'cpu',
                 save_dir: str = 'pool'):
        """
        训练池初始化
        
        Args:
            board_size: 棋盘大小
            initial_size: 初始种群大小
            min_pool_size: 最小种群大小（低于此值时添加新成员）
            max_pool_size: 最大种群大小（高于此值时停止繁殖）
            device: 训练设备
            save_dir: 存储目录
        """
        self.board_size = board_size
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.device = device
        self.save_dir = save_dir
        
        # 创建存储目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化日志
        self.logger = self._setup_logging()
        
        # 初始化种群
        self.members: Dict[str, Mask] = {}
        self._initialize_population(initial_size)
        
        # 统计信息
        self.current_cycle = 0
        self.stats = {
            'population_size': [],
            'average_health': [],
            'average_age': [],
            'best_performance': []
        }
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'pool_{timestamp}.log')
        
        logger = logging.getLogger('TrainPool')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _initialize_population(self, size: int):
        """初始化种群"""
        self.logger.info(f"Initializing population with {size} members")
        for _ in range(size):
            mask = Mask(self.board_size, self.device)
            self.members[mask.id] = mask
            
    def run_cycle(self):
        """运行一个完整的夜昼循环"""
        self.current_cycle += 1
        self.logger.info(f"Starting cycle {self.current_cycle}")
        
        # 消化阶段
        self.logger.info("Digestion phase started")
        for mask in list(self.members.values()):
            if not mask.digest():
                self._remove_member(mask.id)
                
        # 对战阶段
        self.logger.info("Battle phase started")
        battle_pairs = self._generate_battle_pairs()
        for mask1_id, mask2_id in battle_pairs:
            if mask1_id in self.members and mask2_id in self.members:
                mask1 = self.members[mask1_id]
                mask2 = self.members[mask2_id]
                mask1.battle(mask2)
                
                # 检查是否有死亡
                if not mask1.is_alive():
                    self._remove_member(mask1.id)
                if not mask2.is_alive():
                    self._remove_member(mask2.id)
                    
        # 繁殖阶段
        if len(self.members) < self.max_pool_size:
            self.logger.info("Reproduction phase started")
            self._handle_reproduction()
            
        # 年龄更新
        for mask in self.members.values():
            mask.update_age()
            
        # 更新统计信息
        self._update_stats()
        
        # 保存状态
        if self.current_cycle % 10 == 0:
            self.save_state()
            
    def _generate_battle_pairs(self) -> List[Tuple[str, str]]:
        """生成对战配对"""
        available_ids = list(self.members.keys())
        random.shuffle(available_ids)
        
        # 确保每个成员至少进行一次对战
        pairs = []
        while len(available_ids) >= 2:
            id1 = available_ids.pop()
            id2 = available_ids.pop()
            pairs.append((id1, id2))
            
        # 如果还有成员想对战，随机配对
        extra_battles = []
        all_ids = list(self.members.keys())
        num_extra = random.randint(0, len(all_ids) // 2)
        
        for _ in range(num_extra):
            if len(all_ids) >= 2:
                id1, id2 = random.sample(all_ids, 2)
                extra_battles.append((id1, id2))
                
        return pairs + extra_battles
        
    def _handle_reproduction(self):
        """处理繁殖逻辑"""
        # 选择表现最好的成员进行繁殖
        candidates = sorted(
            self.members.values(),
            key=lambda x: x.performance,
            reverse=True
        )[:self.min_pool_size]
        
        # 尝试繁殖
        for parent in candidates:
            if len(self.members) >= self.max_pool_size:
                break
                
            child = parent.reproduce()
            if child:
                self.members[child.id] = child
                self.logger.info(f"New member born: {child.id} (Parent: {parent.id})")
                
        # 如果种群仍然太小，添加新成员
        while len(self.members) < self.min_pool_size:
            new_mask = Mask(self.board_size, self.device)
            self.members[new_mask.id] = new_mask
            self.logger.info(f"Added new member: {new_mask.id}")
            
    def _remove_member(self, member_id: str):
        """移除成员"""
        if member_id in self.members:
            mask = self.members.pop(member_id)
            self.logger.info(f"Member died: {member_id} "
                           f"(Age: {mask.age}, Performance: {mask.performance:.2f})")
            
    def _update_stats(self):
        """更新统计信息"""
        if not self.members:
            return
            
        current_stats = {
            'population_size': len(self.members),
            'average_health': sum(m.health for m in self.members.values()) / len(self.members),
            'average_age': sum(m.age for m in self.members.values()) / len(self.members),
            'best_performance': max(m.performance for m in self.members.values())
        }
        
        for key, value in current_stats.items():
            self.stats[key].append(value)
            
        self.logger.info(f"Cycle {self.current_cycle} stats: "
                        f"Population={current_stats['population_size']}, "
                        f"Avg Health={current_stats['average_health']:.1f}, "
                        f"Avg Age={current_stats['average_age']:.1f}, "
                        f"Best Performance={current_stats['best_performance']:.2f}")
                        
    def save_state(self):
        """保存当前状态"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.save_dir, f'cycle_{self.current_cycle}')
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存统计信息
        stats_file = os.path.join(save_dir, 'stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
            
        # 保存所有成员
        for mask_id, mask in self.members.items():
            model_file = os.path.join(save_dir, f'mask_{mask_id}.pt')
            mask.save(model_file)
            
        self.logger.info(f"State saved to {save_dir}")
        
    def load_state(self, cycle_dir: str):
        """加载状态"""
        if not os.path.exists(cycle_dir):
            raise ValueError(f"Directory not found: {cycle_dir}")
            
        # 加载统计信息
        stats_file = os.path.join(cycle_dir, 'stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
                
        # 加载所有成员
        self.members.clear()
        for file in os.listdir(cycle_dir):
            if file.startswith('mask_') and file.endswith('.pt'):
                mask_id = file[5:-3]  # Remove 'mask_' and '.pt'
                mask = Mask(self.board_size, self.device)
                mask.load(os.path.join(cycle_dir, file))
                self.members[mask_id] = mask
                
        self.logger.info(f"State loaded from {cycle_dir}")
        
    def get_best_member(self) -> Optional[Mask]:
        """获取表现最好的成员"""
        if not self.members:
            return None
            
        return max(self.members.values(), key=lambda x: x.performance)
        
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()