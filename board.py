from typing import List, Tuple

class Board:
    def __init__(self, size, n):
        if size < n:
            raise ValueError("棋盘大小必须大于等于连子数")
        self.size = size
        self.n = n
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.winner = None
        self.last_move = None  # 记录最后一次落子位置：(row, col, player)

    def __str__(self):
        """简化的棋盘显示，使坐标与内部表示一致"""
        result = []
        result.append("  " + " ".join(str(i+1) for i in range(self.size)))
        for i in range(self.size):
            row_str = f"{i+1} " + " ".join(self.board[i])
            result.append(row_str)
        
        # 添加最后落子信息
        if self.last_move:
            row, col, player = self.last_move
            result.append(f"\nLast move: Player {player} at ({row+1}, {col+1})")
        
        return "\n".join(result)

    def is_valid_move(self, row, col):
        """使用内部坐标（从0开始）"""
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == ' '

    def _count_consecutive(self, row, col, player, dr, dc):
        """
        在指定方向上计算连续棋子数
        
        Args:
            row, col: 起始位置
            player: 玩家标记
            dr, dc: 方向增量
        
        Returns:
            连续棋子数量，包括起始位置
        """
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc
        
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        
        return count

    def _count_open_consecutive(self, row, col, player, dr, dc):
        """
        计算特定方向上的开放连续棋子数
        
        Returns:
            (连续棋子数, 是否完全开放)
        """
        count = 1
        r, c = row + dr, col + dc
        open_left = True  # 左侧是否开放
        open_right = True  # 右侧是否开放
        
        # 向一个方向计数
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            elif self.board[r][c] == ' ':
                break
            else:
                open_right = False
                break
        
        # 重置位置并向另一个方向计数
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            elif self.board[r][c] == ' ':
                break
            else:
                open_left = False
                break
        
        return count, open_left and open_right

    def _check_forbidden_for_x(self, row, col):
        """
        检查黑方（X）落子的禁手
        
        Returns:
            是否为禁手，如果是禁手返回True
        """
        # 长连禁手检查
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 长连检查
        for dr, dc in directions:
            if self._count_consecutive(row, col, 'X', dr, dc) > self.n:
                return True
        
        # 双活三检查
        active_three_count = 0
        for dr, dc in directions:
            count, is_open = self._count_open_consecutive(row, col, 'X', dr, dc)
            if count == self.n - 2 and is_open:
                active_three_count += 1
        
        if active_three_count >= 2:
            return True
        
        # 双四检查
        four_count = 0
        for dr, dc in directions:
            count = self._count_consecutive(row, col, 'X', dr, dc)
            if count == self.n - 1:
                four_count += 1
        
        if four_count >= 2:
            return True
        
        return False

    def make_move(self, row, col, player):
        """直接使用内部坐标，不减1"""
        if not self.is_valid_move(row, col):
            return False
        
        # 黑方（X）需要检查禁手
        if player == 'X':
            # 临时落子
            self.board[row][col] = player
            
            # 检查是否为禁手
            if self._check_forbidden_for_x(row, col):
                # 回滚落子并判负
                self.board[row][col] = ' '
                self.winner = 'O'  # 黑方因禁手判负
                return False
        
        # 正常落子
        self.board[row][col] = player
        self.last_move = (row, col, player)
        return True

    def check_win(self, row, col, player):
        """
        检查是否获胜（内部坐标，无需修改）
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.n:
                self.winner = player
                return True
        return False

    def get_last_move(self):
        """
        获取最后一次落子信息
        
        Returns:
            tuple: (row, col, player) 或 None（如果还没有落子）
        """
        return self.last_move

    def has_winner(self):       
        return self.winner is not None

    def is_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def reset(self):
        self.board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.winner = None
        self.last_move = None  # 重置最后落子记录
        
    def get_board(self):
        return [row[:] for row in self.board]
        
    def get_winner(self):
        return self.winner

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """获取所有合法移动"""
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves