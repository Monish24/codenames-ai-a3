# Codenames AI

This project implements an AI framework for the game Codenames with multiple agent strategies, a comprehensive GUI, and tournament systems with TrueSkill ranking.

## Prerequisites

**Python 3.8+ required**

Install dependencies:
```bash
pip install numpy scikit-learn sentence-transformers trueskill pandas matplotlib
```

## Quick Start

1. **Extract and navigate to the project:**
   ```bash
   unzip codenames.zip
   cd codenames
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy scikit-learn sentence-transformers trueskill pandas matplotlib
   ```

3. **Launch the GUI (Recommended):**
   ```bash
   python codenames_gui.py
   ```

## How to Use

### Single Game
**GUI Method (Recommended):**
- Select agents for Red/Blue teams
- Set optional seed
- Click "Start Game"
- Watch real-time gameplay

**Command Line:**
```bash
python run_game.py
```

### Tournaments
**GUI Method (Recommended):**
- Click "Enhanced Tournament"
- Choose Performance or Believability tournament
- Configure agents and settings
- Monitor live progress

**Command Line:**
```bash
python run_believability_tournament.py
```

## Available AI Agents

- **MCTS**: Monte Carlo Tree Search
- **EMD**: Word Embeddings
- **SBERT**: Sentence Transformers  
- **CL**: Curriculum Learning
- **TOT**: Tree of Thoughts
- **Naive**: Simple baseline

## Project Structure

```
codenames/
├── players/           # AI agent implementations
├── results/          # Single game results
├── tournament_results/ # Tournament data
├── codenames_gui.py  # Main GUI application
├── run_game.py       # Single game runner
└── run_believability_tournament.py # Tournament runner
```

## Features

- Interactive GUI with real-time board visualization
- Multiple AI strategies and agent types
- TrueSkill ranking system for tournaments
- Believability analysis for clue quality
- Performance analytics and data export
- Spymaster/Player view toggle

## Results

All games and tournaments automatically save results with detailed analytics including win rates, performance metrics, and statistical analysis.
