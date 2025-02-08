# main.py
import argparse
import torch
from trainloop import train
from ui import main as ui_main

def main():
    parser = argparse.ArgumentParser(description="AlphaGomoku：训练与人机对弈")
    parser.add_argument("--mode", choices=["train", "play"], default="train", help="选择运行模式：train 训练；play 人机对弈")
    args = parser.parse_args()

    # 自动选择设备：如果有 GPU（ROCm），则使用 GPU
    device = torch.device("cpu")
    
    if args.mode == "train":
        train(device=device)
    else:
        ui_main(device=device)

if __name__ == "__main__":
    main()
