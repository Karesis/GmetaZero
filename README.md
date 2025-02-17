# GmetaZero

GmetaZero is an AI-powered Gomoku (Five in a Row) implementation that uses evolutionary algorithms and deep learning to create and improve game-playing agents.

## Features

- Evolutionary training system with multiple AI agents (Masks)
- Neural network-based decision making with ResNet architecture
- Path-based move encoding for better strategic understanding
- Support for standard Gomoku rules including forbidden moves for black pieces
- Interactive gameplay against trained AI agents
- Comprehensive training statistics and state saving/loading

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm
- logging

## Project Structure

- `board.py`: Implementation of the Gomoku game board and rules
- `brain.py`: Neural network architecture for AI decision making
- `fbtree.py`: Fiber tree implementation for move sequence tracking
- `mask.py`: AI agent implementation with lifecycle management
- `trainpool.py`: Training pool management for evolutionary learning
- `test.py`: Main script for training and playing against AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karesis/GmetaZero.git
cd GmetaZero
```

2. Install dependencies:
```bash
pip install torch numpy tqdm
```

## Usage

### Training Mode

To start training the AI:

```bash
python test.py --mode train --board-size 10 --cycles 100 --initial-size 20
```

Parameters:
- `--board-size`: Size of the game board (default: 10)
- `--cycles`: Number of evolution cycles (default: 100)
- `--initial-size`: Initial population size (default: 20)
- `--min-size`: Minimum population size (default: 10)
- `--max-size`: Maximum population size (default: 30)
- `--device`: Training device ('cpu' or 'cuda', default: 'cuda' if available)

### Play Mode

To play against a trained AI:

```bash
python test.py --mode play --load-dir /path/to/saved/state --board-size 10
```

## Training System

The training system uses an evolutionary approach where:
1. AI agents (Masks) perform self-play to improve their skills
2. Agents battle against each other to determine fitness
3. Successful agents can reproduce, creating new agents with slightly modified neural networks
4. Agents have health points and age, creating a natural selection environment

## Development Status

This project is under active development. Current focus areas:
- Improving the neural network architecture
- Optimizing the evolutionary parameters
- Enhancing the training efficiency
- Adding support for different board sizes and rule variations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

Karesis

## Acknowledgments

- Thanks to AlphaZero for inspiration in the neural network architecture
- Thanks to my father(he just give me everything I need)
