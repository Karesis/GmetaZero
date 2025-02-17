from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Move:
    value: int
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.value == other.value
    
    def __str__(self):
        return str(self.value)

class Fiber:
    def __init__(self, moves: List[Move], tree=None, parent = None):
        self.moves = moves
        self.next_fibers: List['Fiber'] = []
        self.visit_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.tree = tree
        self.parent = parent  # 添加父节点引用
    
    def find_matching_next(self, moves: List[Move], start_idx: int = 0) -> Optional['Fiber']:
        """找到匹配给定moves序列的下一个fiber"""
        remaining_moves = moves[start_idx:]
        for next_fiber in self.next_fibers:
            if len(next_fiber.moves) >= len(remaining_moves):
                if next_fiber.moves[:len(remaining_moves)] == remaining_moves:
                    return next_fiber, len(remaining_moves)
            else:
                if remaining_moves[:len(next_fiber.moves)] == next_fiber.moves:
                    return next_fiber, len(next_fiber.moves)
        return None, 0

class FiberTree:
    def __init__(self):
        self.root = Fiber([], self)
        self.current_fiber = self.root
        self.current_moves: List[Move] = []
        self.move_index = 0
        self.adding_mode = False
        
    def start_adding_mode(self):
        self.adding_mode = True
        self.current_fiber = self.root
        self.current_moves = []
        self.move_index = 0
        
    def end_adding_mode(self):
        self.adding_mode = False
        
    def add_move(self, move: Move) -> bool:
        if not self.adding_mode:
            return False
            
        self.current_moves.append(move)
        
        if not self.current_fiber.next_fibers:
            # 如果没有后继fiber，直接创建新的
            new_fiber = Fiber([move], self)
            self.current_fiber.next_fibers.append(new_fiber)
            self.current_fiber = new_fiber
            return True
            
        # 尝试找到匹配的后继fiber
        next_fiber, match_len = self.current_fiber.find_matching_next(self.current_moves, self.move_index)
        
        if next_fiber:
            # 找到匹配的fiber，继续使用
            self.current_fiber = next_fiber
            self.move_index += match_len
        else:
            # 没找到匹配的，创建新分支
            new_fiber = Fiber([move], self)
            self.current_fiber.next_fibers.append(new_fiber)
            self.current_fiber = new_fiber
            self.move_index += 1
            
        return True
        
    def update_statistics(self, win: bool):
        """更新统计信息"""
        fiber = self.current_fiber
        while fiber:
            fiber.visit_count += 1
            if win:
                fiber.win_count += 1
            else:
                fiber.loss_count += 1
            for move in fiber.moves:
                if move in self.current_moves:
                    break
            fiber = None  # 暂时简化统计更新
            
    def find_common_prefix(self, fibers: List['Fiber']) -> List[Move]:
        """找到一组fiber中共同的move前缀"""
        if not fibers:
            return []
        
        # 获取所有moves序列
        all_moves = [fiber.moves for fiber in fibers]
        min_len = min(len(moves) for moves in all_moves)
        
        # 找到共同前缀
        common_prefix = []
        for i in range(min_len):
            current_move = all_moves[0][i]
            if all(moves[i] == current_move for moves in all_moves):
                common_prefix.append(current_move)
            else:
                break
        return common_prefix

    def print_simple_tree(self, fiber: Optional[Fiber] = None, prefix: str = "", depth: int = 0):
        """简单的树形打印，合并共同前缀"""
        if fiber is None:
            print("Tree Structure:")
            fiber = self.root
            if not fiber.next_fibers:
                print("Empty tree")
                return
            # 从第一个非空fiber开始打印
            self.print_simple_tree(fiber.next_fibers[0], "", 0)
            return
            
        # 获取当前路径上所有需要处理的fiber
        current_fibers = [fiber]
        while len(current_fibers[-1].next_fibers) == 1:
            current_fibers.append(current_fibers[-1].next_fibers[0])
            
        # 收集所有moves
        all_moves = []
        for f in current_fibers:
            all_moves.extend(f.moves)
            
        # 打印当前路径
        if all_moves:
            moves_str = ",".join(str(m) for m in all_moves)
            if depth == 0:
                print(f"└──{moves_str}")
            else:
                print(f"{prefix}└─{moves_str}")
            
        # 如果最后一个fiber有多个分支，打印它们
        last_fiber = current_fibers[-1]
        if len(last_fiber.next_fibers) > 1:
            new_prefix = "    " * (depth + 1)
            for next_fiber in last_fiber.next_fibers:
                self.print_simple_tree(next_fiber, new_prefix, depth + 1)
            
    def get_current_path(self) -> List[Move]:
        return self.current_moves
    
    def get_complete_path(self) -> List[Move]:
        """
        获取从根节点到当前节点的完整落子序列
        
        Returns:
            List[Move]: 完整的落子序列
        """
        complete_moves = []
        fiber = self.current_fiber
        while fiber and fiber != self.root:
            complete_moves = fiber.moves + complete_moves
            fiber = fiber.parent
        return complete_moves
    def print_internal_structure(self, fiber: Optional[Fiber] = None, depth: int = 0):
        """打印树的内部结构，用于调试"""
        if fiber is None:
            print("\nInternal Tree Structure:")
            fiber = self.root
        
        indent = "  " * depth
        moves_str = [str(m) for m in fiber.moves]
        print(f"{indent}Fiber(moves={moves_str}, visit={fiber.visit_count})")
        
        for next_fiber in fiber.next_fibers:
            self.print_internal_structure(next_fiber, depth + 1)

def test_fiber_tree():
    tree = FiberTree()
    
    # Test 1: 创建主路径
    print("\n=== Test 1: Main Path ===")
    tree.start_adding_mode()
    moves = [Move(1), Move(2), Move(3), Move(4), Move(5)]
    print("Adding:", ",".join(str(m) for m in moves))
    for move in moves:
        tree.add_move(move)
    tree.end_adding_mode()
    tree.update_statistics(win=True)
    
    print("\nPretty print:")
    tree.print_simple_tree()
    tree.print_internal_structure()
    
    # Test 2: 创建分支
    print("\n=== Test 2: Branch Path ===")
    tree.start_adding_mode()
    moves = [Move(1), Move(2), Move(3), Move(6)]
    print("Adding:", ",".join(str(m) for m in moves))
    for move in moves:
        tree.add_move(move)
    tree.end_adding_mode()
    tree.update_statistics(win=False)
    
    print("\nPretty print:")
    tree.print_simple_tree()
    tree.print_internal_structure()
    
    # Test 3: 另一个早期分支
    print("\n=== Test 3: Early Branch ===")
    tree.start_adding_mode()
    moves = [Move(1), Move(2), Move(7)]
    print("Adding:", ",".join(str(m) for m in moves))
    for move in moves:
        tree.add_move(move)
    tree.end_adding_mode()
    tree.update_statistics(win=True)
    
    print("\nPretty print:")
    tree.print_simple_tree()
    tree.print_internal_structure()

if __name__ == "__main__":
    test_fiber_tree()