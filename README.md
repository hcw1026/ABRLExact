# xxx

This repository contains the code for xxx, including implementations of Bayes-BR, HMC, and Bayes-TD-based methods.

## Methods

The core algorithms are implemented via the following main functions:
* `run_cdf_deepsea`: Implements the **Bayes-BR** method.
* `run_cdf_deepsea_bs`: Implements the Bayes-TD-based methods. The specific variant is selected via the `bootstrap_mode` argument:
  * `0` corresponds to **Bayes-TD-Max**
  * `1` corresponds to **Bayes-TD**
  * `2` corresponds to **Bayes-TD-En**
* `run_hmc_deepsea`: Implements the **HMC** method.

## Installation

1. Clone this repository:
   ```bash
   git clone xxx
   cd ABRLExact
   ```

2. Create a virtual environment and install the required dependencies specified in the `pyproject.toml` file.

   **Using uv:**
   ```bash
   uv sync
   
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

   **Using pip:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -e .
   ```

## Generating Data

Experiments are configured using YAML files (assumed to be located in a `configs/` directory). You can run experiments in parallel using `generate_data.py`.

To run a single configuration:
```bash
python generate_data.py configs/config_deepsea_5x5_det.yaml \
    --base_dir ./data \
    --n_jobs 4 \
    --num_experiments 10 \
    --batch_size 1024
```
* `--n_jobs`: The number of CPU cores for parallel processing to repeat the experiments, one CPU for one repeat.
* `--base_dir`: The directory where the experiment output (data) will be saved.
* `--num_experiments`: The number of repeated experiments to run.
* `--batch_size`: The vectorisation batch size / dimension for Bayes-BR (for memory management purposes only)

### Bulk Data Generation

To generate the data for all the experiments for plotting, use the bash script. (Alternatively, amend the list of configuration files in the script to run specific experiments.)
```bash
bash generate_data.sh
```

## Reproducing Plotting

Once the data has been generated and saved into the `./data` directory, you can reproduce the experiments and generate the plots from the paper.

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `generate_plot.ipynb` in your browser.
3. Run all cells to process the saved simulation data and output the figures.
