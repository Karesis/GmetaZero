# train_loop.py
import torch
import torch.optim as optim
import numpy as np
from nn import AlphaGomokuNet  # 请确保你的神经网络代码保存在 network.py 中
from mcts import MCTS, state_to_tensor
from game import GomokuGame

def self_play_episode(net, num_simulations, device):
    """
    利用当前网络进行自对弈，每一步利用 MCTS 搜索选取落子，
    收集形如 (状态, MCTS 产生的落子概率分布, 当前玩家) 的训练样本。
    """
    game = GomokuGame(board_size=15)
    examples = []
    while not game.is_terminal():
        mcts = MCTS(net, num_simulations=num_simulations, device=device)
        move_probs = mcts.search(game)
        state_tensor = state_to_tensor(game).to(device)
        examples.append((state_tensor, move_probs, game.current_player))
        # 根据 MCTS 输出的概率分布采样选择落子
        moves, probs = zip(*move_probs.items())
        probs = np.array(probs)
        chosen_index = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen_index]
        game.make_move(chosen_move)
    winner = game.get_winner()
    training_examples = []
    for state, pi, player in examples:
        outcome = 0
        if winner is not None:
            outcome = 1 if winner == player else -1
        training_examples.append((state, pi, outcome))
    return training_examples

def train(num_episodes=1000, num_simulations=50, batch_size=32, lr=0.001, device='cpu'):
    """
    训练循环：
      1. 进行 num_episodes 盘自对弈，收集训练数据
      2. 每当积累到一个 batch 时，利用均方误差和 KL 散度损失更新网络参数
    """
    net = AlphaGomokuNet()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn_policy = torch.nn.KLDivLoss()   # 网络输出 log softmax，目标为概率分布
    loss_fn_value = torch.nn.MSELoss()

    training_data = []
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        examples = self_play_episode(net, num_simulations, device)
        training_data.extend(examples)

        if len(training_data) >= batch_size:
            net.train()
            np.random.shuffle(training_data)
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                # 拼接状态张量，尺寸：(batch_size, channels, board_size, board_size)
                states = torch.cat([x[0] for x in batch]).to(device)
                board_size = net.board_size
                # 构造目标落子概率分布张量，尺寸：(batch_size, board_size^2+1)
                target_pi = []
                for _, pi, _ in batch:
                    target = np.zeros(board_size * board_size + 1, dtype=np.float32)
                    for move, prob in pi.items():
                        i_move, j_move = move
                        index = i_move * board_size + j_move
                        target[index] = prob
                    target_pi.append(target)
                target_pi = torch.tensor(target_pi, device=device)
                target_value = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device).unsqueeze(1)

                optimizer.zero_grad()
                out_log_pi, out_value = net(states)
                loss_policy = loss_fn_policy(out_log_pi, target_pi)
                loss_value = loss_fn_value(out_value, target_value)
                loss = loss_policy + loss_value
                loss.backward()
                optimizer.step()
            print(f"Loss: {loss.item():.4f}")
            training_data = []  # 每轮训练后清空数据
    # 保存训练好的模型参数
    torch.save(net.state_dict(), "alphagomoku_net.pth")
    print("模型保存到 alphagomoku_net.pth")

if __name__ == "__main__":
    # 自动选择设备：如果有 GPU（ROCm），则使用 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 示例：训练 10 盘对弈，每盘模拟 20 次 MCTS，batch size 为 16
    train(num_episodes=10, num_simulations=20, batch_size=16, lr=0.001, device=device)
