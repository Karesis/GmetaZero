import os
import torch
from trainpool import TrainPool
from mask import Mask, set_seed
import argparse
from board import Board

def train_evolution(args):
    """运行进化训练"""
    pool = TrainPool(
        board_size=args.board_size,
        initial_size=args.initial_size,
        min_pool_size=args.min_size,
        max_pool_size=args.max_size,
        device=args.device
    )
    
    print(f"\nStarting evolution training for {args.cycles} cycles")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Population: {args.initial_size} (min: {args.min_size}, max: {args.max_size})")
    print(f"Device: {args.device}\n")
    
    try:
        for cycle in range(args.cycles):
            pool.run_cycle()
            
            # 每10个周期显示一次详细统计
            if (cycle + 1) % 10 == 0:
                stats = pool.stats
                print(f"\nCycle {cycle + 1} Summary:")
                print(f"Population: {stats['population_size'][-1]}")
                print(f"Average Health: {stats['average_health'][-1]:.1f}")
                print(f"Average Age: {stats['average_age'][-1]:.1f}")
                print(f"Best Performance: {stats['best_performance'][-1]:.2%}\n")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    return pool

def play_with_best(pool: TrainPool):
    """与最佳个体对弈"""
    best_mask = pool.get_best_member()
    if not best_mask:
        print("No valid AI found!")
        return
        
    print(f"\nPlaying with best AI (ID: {best_mask.id})")
    print(f"Performance: {best_mask.performance:.2%}")
    print(f"Age: {best_mask.age} cycles")
    print(f"Total games: {best_mask.total_games}")
    
    board = Board(pool.board_size, 5)  # 五子棋规则
    best_mask.start_game()
    moves_history = []
    
    print(f"\nGame started! Board size: {pool.board_size}x{pool.board_size}")
    print("You are O, AI is X. Need 5 in a row to win.")
    print(board)
    
    while not board.is_full() and not board.has_winner():
        # AI回合
        valid_moves = board.get_valid_moves()
        board_state = board.get_board()
        move = best_mask.get_move(board_state, valid_moves, 'X', temperature=0.1)
        
        if move and board.make_move(move[0], move[1], 'X'):
            moves_history.append((move[0], move[1], 'X'))
            print(f"\nAI moves to: ({move[0]+1}, {move[1]+1})")
            print(board)
            if board.check_win(move[0], move[1], 'X'):
                print("AI wins!")
                best_mask.end_game(True)
                break
        
        if board.is_full():
            break
            
        # 人类回合
        while True:
            try:
                row = int(input(f"Enter row (1-{pool.board_size}): ")) - 1
                col = int(input(f"Enter col (1-{pool.board_size}): ")) - 1
                if board.make_move(row, col, 'O'):
                    moves_history.append((row, col, 'O'))
                    break
                print("Invalid move, try again")
            except ValueError:
                print("Please enter valid numbers")
                
        print(board)
        if board.check_win(row, col, 'O'):
            print("You win!")
            best_mask.end_game(False)
            break
            
    if not board.has_winner() and board.is_full():
        print("Draw!")
        best_mask.end_game(None)
    
    print("\nGame moves:")
    for i, (row, col, player) in enumerate(moves_history, 1):
        print(f"{i}. {player} at ({row+1}, {col+1})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolution Training and Play')
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                      help='Mode: train the AI or play against it')
    parser.add_argument('--board-size', type=int, default=10,
                      help='Board size')
    parser.add_argument('--cycles', type=int, default=100,
                      help='Number of evolution cycles')
    parser.add_argument('--initial-size', type=int, default=20,
                      help='Initial population size')
    parser.add_argument('--min-size', type=int, default=10,
                      help='Minimum population size')
    parser.add_argument('--max-size', type=int, default=30,
                      help='Maximum population size')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--load-dir', type=str, default=None,
                      help='Directory to load saved state from')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    if args.mode == 'train':
        # 训练模式
        pool = train_evolution(args)
        
        # 训练完成后询问是否要玩游戏
        print("\nEvolution training completed!")
        answer = input("Would you like to play against the best AI? (y/n): ")
        if answer.lower() == 'y':
            play_with_best(pool)
            
    else:  # play mode
        if not args.load_dir:
            print("Please specify a saved state directory with --load-dir")
            exit(1)
            
        # 加载已有的训练池
        pool = TrainPool(
            board_size=args.board_size,
            initial_size=args.initial_size,
            min_pool_size=args.min_size,
            max_pool_size=args.max_size,
            device=args.device
        )
        
        try:
            pool.load_state(args.load_dir)
            play_with_best(pool)
        except Exception as e:
            print(f"Error loading state: {e}")
            exit(1)
            
    print("\nThanks for playing!")