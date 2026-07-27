# Asynchronous Advantage Actor-Critic (A3C) Reinforcement Learning for Multi-Agent Path Planning with Expert-Based Imitation Learning

This repository contains the implementation and experimental results of a decentralized learning framework for multi-agent pathfinding (MAPF).

The proposed approach addresses the scalability limitations of centralized planners by enabling agents to learn coordinated navigation policies using only local observations. Our method combines reinforcement learning (RL) and imitation learning (IL) in a hybrid training framework based on Asynchronous Advantage Actor-Critic (A3C).

A deep neural network (DNN) acts as the policy model and is trained asynchronously using filtered expert demonstrations to discourage goal-blocking behaviors and improve cooperative navigation among agents.

The trained policy is evaluated in multiple simulation environments, demonstrating fast convergence, high success rates, and effective collision avoidance, while maintaining strong scalability in decentralized multi-agent settings.

## Reproducing Tables and Figures from the Manuscript

The following table links each tables and figures presented in the manuscript to the corresponding **output/results directories** and **Jupyter notebooks** used to generate them.

| Figure/Table No. | Output / Results Directory         | Notebook / Script                                                    |
| ---------------- | ---------------------------------- | -------------------------------------------------------------------- |
| Table 1          | [paper_results_tables/table1](./paper_results_tables/Table_1/Results_table_1) | [paper_results_tables/table1_analysis.ipynb](./paper_results_tables/Table_1/Table_1_analysis.ipynb) |
| Table 2          | [paper_results_tables/table2](./paper_results_tables/Table_2/Results_table_2) | [paper_results_tables/table2_analysis.ipynb](./paper_results_tables/Table_2/Table_2_analysis.ipynb) |
| Table 3          | [paper_results_tables/table3](./paper_results_tables/Table_3/Results_table_3) | [paper_results_tables/table3_analysis.ipynb](./paper_results_tables/Table_3/Table_3_analysis.ipynb) |
| Table 4          | [paper_results_tables/table4](./paper_results_tables/Table_4/Results_table_4) | [paper_results_tables/table4_analysis.ipynb](./paper_results_tables/Table_4/Table_4_analysis.ipynb) |
| Table 5          | [paper_results_tables/table5](./paper_results_tables/Table_5/Results_table_5) | [paper_results_tables/table5_analysis.ipynb](./paper_results_tables/Table_5/Table_5_analysis.ipynb) |
| Figure 6 | [images/fixed_world_1](./Experimental_results_of_Different_models/fixed_structured_grid_world_test_videos/fixed_structured_grid_world_1/fixed_structured_grid_world_1_agents_50_size_70.mp4) | Visualization from structured world 1 evaluation |
| Figure 7 | [images/fixed_world_2](./Experimental_results_of_Different_models/fixed_structured_grid_world_test_videos/fixed_structured_grid_world_2/fixed_structured_grid_world_2_agents_100_size_70.mp4) | Visualization from structured world 2 evaluation |
| Figure 8 | [images/fixed_world_3](./Experimental_results_of_Different_models/fixed_structured_grid_world_test_videos/fixed_structured_grid_world_3/fixed_structured_grid_world_3_agents_100_size_70.mp4) | Visualization from structured world 3 evaluation |

### Project Structure Overview

| Folder / File                                 | Type    | Description                                                                 |
|-----------------------------------------------|---------|-----------------------------------------------------------------------------|
| `od_mstar3/`                                  | Folder  | Contains the expert planner implementation based on M* algorithm. |
| `main.py`                                     | Script  | Starts training using A3C and imitation learning.                  |
| `evaluate_a3c_network.py`                     | Script  | Evaluates trained models on fixed benchmark environments.                   |
| `Experimental_results_of_Different_models/`   | Folder  | Contains performance results of each model across test scenarios.          |
| `training_logs_and_trained_models/`           | Folder  | Includes training logs and saved models from training runs.                |
| `PRIMAL/`                                     | Folder  | Original TensorFlow implementation code of the PRIMAL path planning model. |
| `fixed_benchmark_environments/`               | Folder  | Predefined scenarios for consistent evaluation across agents.              |
---

## Setup

### 1. Compile Expert Planner (`cpp_mstar`)
Before training or imitation learning, compile the C++ M* path planner:
cd into the od_mstar3 folder.
```bash
cd od_mstar3
python3 setup.py build_ext --inplace
```
Then copy the generated .so file from the build/lib.*/ directory into the root of the od_mstar3 folder:
```bash
cp build/lib.*/cpp_mstar*.so .
```
Test the import by going back to the repository root and running: python3 and "import cpp_mstar"
```bash
cd ..
python3 -c "import cpp_mstar"
```
### 2. Install Dependencies

Install all required dependencies:

```bash
pip  torch==1.12.1 numpy==1.24.3 scipy==1.13.1 tensorboard==2.18.0 tensorboardX==2.6.2.2 setproctitle==1.3.4 PyYAML==6.0.2 Cython matplotlib imageio networkx
```

---

## Running Training

```bash
python3 main.py
```

Training dynamically switches between imitation learning (from ODrM\*) and purely reinforcement learning depending on the episode configuration.

Logs:

* Scalar logs: `training_logs.txt`
* TensorBoard: `training_logs/`

---

### After Training

To Evaluate the trained model:

```bash
python evaluate_a3c_network.py
```

Use the `fixed_benchmark_environments/` folder to load fixed scenarios with different complexity.

## How to Run Inference on Original PRIMAL Model in Our Benchmark World

To evaluate the original PRIMAL model within our custom multi-agent pathfinding benchmark environment, follow the steps below:

### Step 1: Clone the PRIMAL Repository

Clone the original PRIMAL repository into a subfolder named `PRIMAL_original` inside your working directory:

```bash
git clone <https://github.com/gsartoretti/PRIMAL.git> PRIMAL
```
### Step 2:  Install Dependencies
Follow the instructions in their PRIMAL README to install the required dependencies.

### Step 3: Use Our Benchmark World

We provide fixed benchmark environments for inference:

- Use the folder `fixed_benchmark_environments/` to access pre-generated worlds for evaluation.

- You can also generate new randomized environments using the provided script:

```bash
python generate_environments.py
```

### Step 4: Run PRIMAL Inference

After setting up everything, run the inference using the script provided in our project:

```bash
python primal_testing_my_world.py
```


## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{noushad2026a3c_mapf,
  title   = {Asynchronous Advantage Actor-Critic (A3C) Reinforcement Learning for Multi-Agent Path Planning with Expert-Based Imitation Learning},
  author  = {Mohamed Farhan Kunjanamkattil Noushad and Sandesh Athni Hiremath and Nicolas R. Gauger},
  institution  = {RPTU Kaiserslautern-Landau},
  year    = {2026}
}
```

## Paper

A conference paper of this work is currently under preparation and shall be soon made available.
