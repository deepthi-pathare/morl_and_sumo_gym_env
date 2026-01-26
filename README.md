# SUMO-based autonomous truck driving environment and Multi-Objective Reinforcement Learning (MORL) framework based on PPO.

## Overview

This repository provides the implementation of Multi-Objective PPO with GPI-LS algorithm. It also includes a custom RL environment for highway driving
for autonomous trucks suing SUMO. Environments tailored for single-objective RL and multi-objective RL are provided.

The goal is to study trade-offs between multiple conflicting objectives such as safety,
energy efficiency, and time efficiency in realistic traffic simulations.

## Repository Structure

    morl/                   # GPILS-MOPPO implementation, and experiments
    truck_driving_rl_env/   # SUMO-based gym-compatible truck driving environment
    requirements.txt        # Python dependencies

## Installation

1.  Clone the repository:

``` bash
cd morl_and_sumo_gym_env
```

2.  Create a virtual environment and install dependencies:

``` bash
pip install -r requirements.txt 
pip install -e truck_driving_rl_env/sumo_gym_env # Environment tailored for single-objective RL
pip install -e truck_driving_rl_env/sumo_gym_env_mo # Environment tailored for multi-objective RL
```

> **Note:** SUMO must be installed separately and available in your
> `PATH`, in order to visualize the simulations.

## Usage
Typical workflow:
1. Configure environment parameters for the experiment in `morl/env_parameters.py`
2. Launch experiments using the provided script in ```morl/launch_experiment.py```
   #### Example
   Run a GPILS-MOPPO experiment in the SUMO highway multi-objective environment:

   ```bash
      python morl/launch_experiment.py \
      --algo gpils_moppo \
      --env-id sumo_highway_env_mo-v0 \
      --num-timesteps 1000000 \
      --gamma 0.98 \
      --ref-point -101 -101 -101 \

