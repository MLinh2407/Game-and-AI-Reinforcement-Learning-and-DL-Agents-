# Reinforcement Learning Project: Gridworld & Arena

**Course:** Games and Artificial Intelligence Techniques  
**Assignment:** Final Project - Reinforcement Learning and Deep Learning Agents

## Overview

This project implements two reinforcement learning environments from scratch:

**Part I - Gridworld (Classical RL):** A tile-based game where an agent learns to navigate, collect items, and avoid hazards using Q-Learning and SARSA algorithms. This demonstrates fundamental value-based reinforcement learning.

**Part II - Arena (Deep RL):** A real time shooting game where an agent learns to survive enemy waves and destroy spawners using deep reinforcement learning (PPO with neural networks). This demonstrates modern deep RL techniques.

Both environments are built with Pygame and provide visual feedback of the learning process.

## Quick Start Guide

### Step 1: Run Part I (Gridworld)
```bash
# Navigate to part 1 directory
cd Part1_Classical_RL

# Run the gridworld environment
python main.py
```

**What you'll see:**
- A 10x10 gridworld with tiles and sprites
- UI buttons to select algorithms (Q-Learning/SARSA) and levels (0-6)
- Real time training statistics and learning curves

**What to do:**
1. Click "Q-Learning" or "SARSA" to select algorithm
2. Click a level button (start with "Level 0")
3. Click "Play / Pause" to start training
4. You can also Load the trained model in that level to view the learned policy agent

### Step 2: Run Part II (Arena)
```bash
# Navigate to part 2 directory
cd Part2_Deep_RL

# Option A: Rotation Control Set
# Step 1: Train a new model
python train_rotation.py

#Step 2: Evaluate a trained model
python evaluate_rotation.py

# Option B: Direction Control Set
# Step 1: Train a new model
python train_directional.py

# Step 2: Evaluate a trained model
python evaluate_directional.py

# Option C: Test the environment manually
python test.py
```

**What you'll see:**
- A space themed shooting arena with enemies and spawners
- Real time gameplay with particles and visual effects
- Training progress (if training) or learned behavior (if evaluating)

**What to do in test.py:**
1. Click "Human Control" to play manually, or let the random agent play
2. Press `H` key to toggle human control on/off
3. Use WASD + Space to control the ship
4. Click "Fast Mode" to speed up simulation
