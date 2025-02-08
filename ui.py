# ui.py
import torch
from mcts import GomokuGame, MCTS, state_to_tensor
from nn import AlphaGomokuNet

def print_board(game):
    board = game.board
    board_size = game.board_size
    for i in range(board_size):
        line = ""
        for j in range(board_size):
            if board[i, j] == 1:
                line += "X "
            elif board[i, j] == -1:
                line += "O "
            else:
                line += ". "
        print(line)
    print()

def human_move(game):
    board_size = game.board_size
    while True:
        try:
            move_input = input("请输入落子坐标（格式 row,col，例如 7,7）：")
            i, j = map(int, move_input.strip().split(','))
            if i < 0 or i >= board_size or j < 0 or j >= board_size:
                print("超出棋盘范围，请重新输入。")
                continue
            if game.board[i, j] != 0:
                print("该位置已有棋子，请重新输入。")
                continue
            return (i, j)
        except Exception as e:
            print("输入格式错误，请按照 row,col 格式输入。")

def ai_move(game, net, num_simulations=50, device='cpu'):
    mcts = MCTS(net, num_simulations=num_simulations, device=device)
    move_probs = mcts.search(game)
    # 选取概率最大的落子
    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    return best_move

def main(device='cpu'):
    net = AlphaGomokuNet()
    try:
        net.load_state_dict(torch.load("alphagomoku_net.pth", map_location=device))
        print("成功加载训练好的模型。")
    except Exception as e:
        print("未找到训练好的模型，将使用未训练的模型。")
    net.to(device)
    
    game = GomokuGame(board_size=15)
    side = input("请选择你执棋的一方（输入 X 表示黑棋，O 表示白棋）：").strip().upper()
    human_player = 1 if side == "X" else -1

    while not game.is_terminal():
        print_board(game)
        if game.current_player == human_player:
            move = human_move(game)
        else:
            print("AI 正在思考……")
            move = ai_move(game, net, num_simulations=50, device=device)
            print(f"AI 落子：{move}")
        game.make_move(move)
    print_board(game)
    winner = game.get_winner()
    if winner is None:
        print("平局！")
    elif winner == human_player:
        print("你赢了！")
    else:
        print("AI 获胜！")

if __name__ == "__main__":
    # 自动选择设备：如果有 GPU（ROCm），则使用 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device=device)
