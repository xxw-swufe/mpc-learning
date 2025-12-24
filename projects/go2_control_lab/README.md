# Go2 Control Laboratory

A project for controlling Unitree Go2 quadruped robot using Isaac Lab.

## Project Structure

```
go2_control_lab/
├── pyproject.toml          # Project configuration
├── README.md              # This file
├── go2_control_lab/       # Main package
│   ├── __init__.py
│   ├── robots/            # Robot models
│   │   ├── __init__.py
│   │   └── go2.py         # Go2 robot wrapper
│   ├── controllers/       # Control algorithms
│   │   ├── __init__.py
│   │   ├── diff_ik.py     # Differential IK
│   │   └── mpc.py         # Model Predictive Control
│   ├── envs/              # Environments
│   │   ├── __init__.py
│   │   └── stand_env.py   # Standing test environment
│   ├── scripts/           # Runnable scripts
│   │   ├── stand_go2.py   # Main entry point
│   │   └── debug_fk_ik.py # Debug utilities
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── frames.py      # Frame transformations
│       └── filters.py     # Digital filters
└── assets/                # Optional assets
```

## Installation

```bash
cd /home/xx/IsaacLab/go2_control_lab
pip install -e .
```

## Usage

```bash
# Run standing test
python -m go2_control_lab.scripts.stand_go2

# Or using the script entry point
stand-go2
```

## TODO

- Implement MPC controller
- Implement differential IK
- Create standing environment
- Add frame transformation utilities
- Add digital filters