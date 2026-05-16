# Nested Learning Validation on MLP based Models
## Proposal

## Project Architecture
```plain
nested_learning_cl/
├── README.md               # Project overview and run instructions
├── requirements.txt        # Core dependencies
├── main.py                 # The CLI entry point for running specific ablation phases
└── src/                    # Main source code directory
    ├── __init__.py
    ├── data/               # Data ingestion and task splitting
    │   ├── __init__.py
    │   └── split_mnist.py
    ├── models/             # Network architectures
    │   ├── __init__.py
    │   ├── baseline.py     # Standard MLP
    │   └── cms_mlp.py      # Continuum Memory System MLP (Stub)
    ├── optimizers/         # Optimizer definitions
    │   ├── __init__.py
    │   ├── factory.py      # Logic to instantiate the correct optimizer
    │   ├── m3.py           # Multi-timescale Momentum Muon (Stub)
    │   └── muon.py         # Standard Muon implementation (Stub)
    └── engine/             # Training loops and evaluation logic
        ├── __init__.py
        ├── metrics.py      # Accuracy Matrix (R), F, BWT calculations
        └── trainer.py      # Task-incremental training loop
```

## Build Environment
> Run the following commands inside the project directory!
```bash
conda create -n nlvm python=3.12 -y
conda activate nlvm
pip install -r requirements.txt
```

## Usage
> make sure to setup the environment first and activate the virtual environment before running the following commands.
```bash
python main.py --model ["cms", "baseline"] --optimizer ["SGD", "M3", "Muon", "Adam"]
```