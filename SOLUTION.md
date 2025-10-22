# Example Solution

## Installation

### Quick Start with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust. It's 10-100x faster than pip.

#### Step 1: Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

#### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/uiuc-ae598-rl/hw2-tabular-solution.git
cd hw2-tabular-solution

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**That's it!** `uv sync` will:
- Create a virtual environment (`.venv/`)
- Install all dependencies from `pyproject.toml`
- Install the project in editable mode
- Lock dependencies for reproducibility

#### Step 3: Verify Installation

```bash
python test_implementation.py
```

### What `uv sync` Does

Unlike `pip install -e .`, `uv sync` provides:
- **Faster installation** - 10-100x faster than pip
- **Dependency locking** - Creates `uv.lock` for reproducible installs
- **Automatic venv** - Creates `.venv/` if it doesn't exist
- **Better resolution** - Smarter dependency conflict handling
- **Sync state** - Installs exactly what's in `pyproject.toml`

### Alternative: Traditional pip

If you prefer pip:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### Running the Code

```bash
# Quick test (2-3 minutes)
python test_implementation.py

# Run everything (30-60 minutes)
python run_all.py

# Or run individual scripts
python scripts/1.train_policies.py      # Train agents
python scripts/2.plot_eval_returns.py   # Plot learning curves
python scripts/3.plot_policy_and_vf.py  # Plot policies/values
```

## Algorithms

### 1. Monte Carlo Control (`hw2/algo/mc.py`)
- On-policy first-visit MC control with epsilon-greedy policy
- Updates Q-values after complete episodes
- Implements epsilon decay for exploration-exploitation balance

### 2. SARSA (`hw2/algo/sarsa.py`)
- On-policy TD(0) control algorithm
- Updates Q-values at each time step using (S, A, R, S', A')
- Uses epsilon-greedy policy for action selection

### 3. Q-Learning (`hw2/algo/q.py`)
- Off-policy TD(0) control algorithm
- Updates Q-values using max Q-value of next state
- Behavior policy is epsilon-greedy, target policy is greedy

## Utilities

### Rendering (`hw2/utils/render.py`)
Functions to visualize:
- Value functions as heatmaps
- Policies as arrow grids
- Combined policy and value function plots

## Scripts

### 1. Training Script (`scripts/1.train_policies.py`)
Trains 10 policies for each configuration:
- 3 algorithms (MC, SARSA, Q-Learning)
- 2 environment settings (is_slippery: True/False)
- Total: 60 trained agents

Features:
- Periodic evaluation every 100 episodes
- Saves evaluation returns and timesteps
- Saves trained agents for later analysis
- Progress bars with tqdm

### 2. Evaluation Returns Plotting (`scripts/2.plot_eval_returns.py`)
Generates plots showing:
- Average returns across 10 runs
- 25th-75th percentile shading
- Separate plots for slippery and non-slippery environments
- Final performance statistics table

### 3. Policy and Value Function Plotting (`scripts/3.plot_policy_and_vf.py`)
Generates visualizations:
- Best agent's policy and value function for each configuration
- Average value functions across all runs
- Individual plots for each algorithm and environment
- Grid showing all 10 runs for consistency analysis

## Key Features

1. **Multiple Runs**: Each configuration is trained 10 times to account for stochasticity
2. **Percentile Intervals**: Plots show 25th-75th percentile intervals for robustness
3. **NumPy-style Docstrings**: All functions documented with parameters, returns, and descriptions
4. **Modular Design**: Algorithms, utilities, and scripts are separated for clarity
5. **Comprehensive Visualization**: Multiple plot types for thorough analysis

## Detailed Usage

### Option 1: Run Everything at Once

```bash
python run_all.py
```

This convenience script will:
1. Run a quick test to verify implementation
2. Train all 60 agents (10 runs × 3 algorithms × 2 settings)
3. Generate all plots automatically
4. Ask to skip training if results already exist

### Option 2: Run Scripts Individually

```bash
# Test implementation (optional, ~2-3 min)
python test_implementation.py

# Train all policies (~30-60 min)
python scripts/1.train_policies.py

# Plot evaluation returns (~1 min)
python scripts/2.plot_eval_returns.py

# Plot policies and value functions (~1 min)
python scripts/3.plot_policy_and_vf.py
```

### Customizing Training

Edit `scripts/1.train_policies.py` to adjust:
- `n_runs = 10` - Number of runs per configuration (line 308)
- `n_episodes = 10000` - Episodes per run (line 309)
- `eval_interval = 100` - Evaluate every N episodes (line 310)

### Tips

- **Reduce training time**: Lower `n_runs` or `n_episodes`
- **Re-plot without retraining**: Just run plotting scripts if `results/` exists
- **Remote servers**: Set `export MPLBACKEND=Agg` for headless plotting

## Output Structure

```
results/
├── MC_slippery.pkl
├── MC_not_slippery.pkl
├── SARSA_slippery.pkl
├── SARSA_not_slippery.pkl
├── Q-Learning_slippery.pkl
└── Q-Learning_not_slippery.pkl

figures/
├── evaluation_returns.png
├── evaluation_returns_slippery.png
├── evaluation_returns_not_slippery.png
├── policy_and_value_slippery.png
├── policy_and_value_not_slippery.png
├── avg_value_functions_slippery.png
├── avg_value_functions_not_slippery.png
└── [algorithm]_[setting]_policy_value.png (6 files)
```

## Hyperparameters

All algorithms use:
- Discount factor: γ = 0.95
- Initial epsilon: 1.0
- Epsilon decay: 0.9995 per episode
- Minimum epsilon: 0.01

TD algorithms (SARSA, Q-Learning):
- Learning rate: α = 0.1

Training:
- Episodes: 10,000 per run
- Evaluation interval: every 100 episodes
- Evaluation episodes: 100 per evaluation

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project directory and venv is activated
cd /path/to/hw2-tabular-solution
source .venv/bin/activate

# Reinstall in editable mode
uv sync --reinstall
# or with pip
pip install -e .
```

### uv Command Not Found
```bash
# Check if uv is in PATH
uv --version

# If not found, restart terminal or add to PATH
source ~/.bashrc  # or ~/.zshrc on macOS
```

### Python Version Issues
```bash
# Check Python version (requires 3.13+)
python --version

# Create venv with specific Python version
uv venv --python 3.13
uv sync
```

### Display Errors on Remote Servers
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
python scripts/2.plot_eval_returns.py
```

Plots will still be saved to `figures/` directory.