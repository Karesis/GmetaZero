import torch

class GomokuGame:
    """
    简单的五子棋游戏状态表示：
      - board：大小为 board_size×board_size 的 numpy 数组，取值 0（空）、1（黑子）、-1（白子）
      - current_player：当前落子方（1 或 -1）
    """
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = torch.zeros((board_size, board_size), dtype=torch.int)
        self.current_player = 1
        self.last_move = None

    def clone(self):
        new_game = GomokuGame(self.board_size)
        new_game.board = self.board.clone
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        return new_game

    def get_legal_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        """
        执行落子，move 为 (i, j) 坐标；这里未处理 pass（弃子）情况。
        """
        if move is None:
            self.last_move = None
        else:
            i, j = move
            if self.board[i, j] != 0:
                raise ValueError("非法落子：该位置已有子")
            self.board[i, j] = self.current_player
            self.last_move = move
        self.current_player = -self.current_player

    def is_terminal(self):
        if self.get_winner() is not None:
            return True
        if torch.all(self.board != 0):
            return True
        return False

    def get_winner(self):
        """
        检查是否有一方连成五子，若有返回对应玩家（1 或 -1），否则返回 None。
        """
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:  # 检查当前位置是否有棋子
                    player = self.board[i, j]
                    # 横向检查
                    if j <= self.board_size - 5 and torch.all(self.board[i, j:j+5] == player):
                        return player
                    # 纵向检查
                    if i <= self.board_size - 5 and torch.all(self.board[i:i+5, j] == player):
                        return player
                    # 对角线（向右下）检查
                    if i <= self.board_size - 5 and j <= self.board_size - 5:
                        if torch.all(self.board[i:i+5, j:j+5].diagonal() == player):
                            return player
                    # 对角线（向右上）检查
                    if i >= 4 and j <= self.board_size - 5:
                        if torch.all(self.board[i:i-5:-1, j:j+5].diagonal() == player):
                            return player
        return None
