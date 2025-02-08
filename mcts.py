# mcts.py
import math
import torch

class MCTSNode:
    """
    MCTS 树中的节点，保存了当前游戏状态、父节点、落子信息及搜索统计量：
      - N：访问次数
      - W：累计评估值
      - P：由神经网络输出的先验概率
    """
    def __init__(self, game_state, parent=None, move=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # 从父状态落到当前状态所采用的落子
        self.children = {}  # { move: MCTSNode }
        self.N = 0
        self.W = 0
        self.P = prior
        self.is_expanded = False

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

def state_to_tensor(game):
    """
    将游戏状态转换为神经网络输入张量：
      输入尺寸：(1, history*2+3, board_size, board_size)
      这里历史信息置 0，仅用后三个平面：
         - 倒数第 3 个平面：当前玩家的棋子（1 表示有棋子，否则 0）
         - 倒数第 2 个平面：对方的棋子
         - 倒数第 1 个平面：空位（1 表示空，0 表示有子）
    """
    board_size = game.board_size
    channels = 8 * 2 + 3  # 若 history=8，则总共 19 个通道
    tensor = torch.zeros((channels, board_size, board_size), dtype=torch.float32)
    tensor[-3] = (game.board == game.current_player).astype(torch.float32)
    tensor[-2] = (game.board == -game.current_player).astype(torch.float32)
    tensor[-1] = (game.board == 0).astype(torch.float32)
    return tensor.unsqueeze(0)  # 添加 batch 维度

class MCTS:
    """
    基于神经网络评估的蒙特卡洛树搜索（MCTS）实现。
    参数：
      - net：神经网络模型，用于评估当前局面（输出 log_policy 与 value）
      - c_puct：探索常数
      - num_simulations：搜索时模拟的次数
      - device：torch 设备（例如 "cuda:0"）
    """
    def __init__(self, net, c_puct=1.0, num_simulations=100, device='cpu'):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device

    def search(self, game_state):
        self.root = MCTSNode(game_state.clone())
        for _ in range(self.num_simulations):
            node = self.root
            # —— 选择阶段 ——
            while node.is_expanded and not node.game_state.is_terminal():
                best_score = -float('inf')
                best_move = None
                for move, child in node.children.items():
                    u = child.Q() + self.c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
                    if u > best_score:
                        best_score = u
                        best_move = move
                if best_move is None:
                    break
                node = node.children[best_move]
            # —— 评估阶段 ——
            if node.game_state.is_terminal():
                winner = node.game_state.get_winner()
                if winner is None:
                    value = 0
                else:
                    # 当局面终止时，返回值按上一步落子的玩家视角取正负
                    value = 1 if winner == -node.game_state.current_player else -1
            else:
                value = self.expand_and_evaluate(node)
            # —— 回传阶段 ——
            self.backpropagate(node, value)
        # 以根节点子节点访问次数构造落子概率分布
        move_visits = {move: child.N for move, child in self.root.children.items()}
        total = sum(move_visits.values())
        move_probs = {move: visits / total for move, visits in move_visits.items()}
        return move_probs

    def expand_and_evaluate(self, node):
        """
        用神经网络扩展当前节点，并返回局面评估值。
        同时对所有合法落子生成子节点，并赋予由神经网络输出（经过 mask 和归一化）的先验概率。
        """
        state_tensor = state_to_tensor(node.game_state).to(self.device)
        self.net.eval()
        with torch.no_grad():
            log_policy, value = self.net(state_tensor)
        policy = torch.exp(log_policy).cpu().numpy().flatten()  # 长度 = board_size^2+1
        legal_moves = node.game_state.get_legal_moves()
        board_size = node.game_state.board_size
        mask = torch.zeros_like(policy)
        for move in legal_moves:
            i, j = move
            index = i * board_size + j
            mask[index] = 1
        policy = policy * mask
        s = torch.sum(policy)
        if s > 0:
            policy /= s
        else:
            policy = mask / torch.sum(mask)
        node.is_expanded = True
        for move in legal_moves:
            i, j = move
            index = i * board_size + j
            prior = policy[index]
            new_game = node.game_state.clone()
            new_game.make_move(move)
            node.children[move] = MCTSNode(new_game, parent=node, move=move, prior=prior)
        return value.item()

    def backpropagate(self, node, value):
        # 将评估值反向传播到根节点，每经过一步则翻转符号
        while node is not None:
            node.N += 1
            node.W += value
            value = -value
            node = node.parent
