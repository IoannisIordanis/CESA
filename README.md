# Counterfactual Explanations for Survival Analysis (CESA)

This repository contains the codebase for generating counterfactual explanations for survival analysis models. It implements four explanation methods:
1. **CESA** (Counterfactual Explainer via Shape preservation and Priority search)
2. **PSO** (Particle Swarm Optimization counterfactual baseline)
3. **MPSO** (Multi-objective Particle Swarm Optimization baseline)
4. **NN** (Nearest Neighbors counterfactual baseline)

---

## 📋 Requirements & Setup

Make sure you have Python 3.8+ installed. The environment installation sequence is as follows:

```bash
pip install numpy pandas scikit-learn mlflow pdmlabs
pip install scipy==1.16.3 --force-reinstall
pip install pycox
```

### Pretrained Models
Place your pretrained pickled models under the `pretrainedmodels/` directory in the repository root. For example:
* `pretrainedmodels/HNEI_CoxPH_model.pkl`

---

## 📁 Project Structure

The project has been organized into modular packages to keep the codebase structured:

* **`CESA/`**: Core Python package containing the CESA algorithm components:
  * `__init__.py`: Exposes `CounterfactualExplainer`.
  * `counterfactual_explainer.py`: Main explainer class coordinating priority & greedy search.
  * `priority_search.py`: Implementation of priority feature search.
  * `greedy_search.py`: Implementation of greedy search backup.
  * `candidate_selection.py`: Logic to choose matching counterfactual donor candidates.
  * `swap_evaluation.py`: Swapping features and validation of shape constraints.
  * `feature_ranking.py`: Distance and permutation-based feature ranking.
  * `evaluation_metrics.py`: Metrics definitions (sparsity, proximity, correlation preservation).
* **`utils/`**: Shared helper package containing:
  * `__init__.py`: Exposes dataset utilities.
  * `utils.py`: Contains data loading, preprocessing, and dataset configurations (e.g., PBC, HNEI).
  * `survival_utils.py`: Survival probability calculation, window flattening, and mathematical utilities.
  * `global_shap.py`: Computes global SHAP importance for a survival model.
  * `local_shap.py`: Computes local SHAP importance for a single instance.
* **Root Directory**:
  * `test_counterfactual.py`: Command line tool for running CESA, PSO, MPSO, and NN experiments.
  * `analyze_results.py`: Command line tool for aggregating and comparing experiment output files.
  * `pso_explainer.py`: Baseline particle swarm optimization counterfactual explainer.
  * `nn_explainer.py`: Baseline nearest-neighbors explainer.
  * `surv_cf_explainer.py`: Implementation of the MPSO explainer baseline.

---

### Data

We use two datasets for benchmarking: 
- HNEI data from [here](https://github.com/ignavinuales/Battery_RUL_Prediction/tree/main/Datasets/HNEI_Processed)
- PBCv2 data from [here](https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets/pbc2.csv)

## 🚀 Running Experiments (`test_counterfactual.py`)

You can run experiments directly from the terminal using `test_counterfactual.py`.

### Command-line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--explainer` | Explainer algorithm to run (`cesa`, `pso`, `nn`, `mpso`). | `cesa` |
| `--dataset` | Name of the dataset to evaluate (`HNEI`, `PBC`, or `SCANIA` variants). | `HNEI` |
| `--model-path` | Filename of the pretrained pickle model. The script automatically looks for it in the `pretrainedmodels/` folder. | `HNEI_CoxPH_model.pkl` |
| `--rho-list` | Comma-separated target times $\tau$ (floats, fractions, or `change_point`). | `0.3333333333333333,0.6666666666666666` |
| `--delta1-list` | Comma-separated delta percentages (floats). | `0.05,0.1,0.2,0.3,0.5` |
| `--max-test-windows` | Limit the number of test windows evaluated (for quick testing). | Run all eligible |
| `--min-rul` | Minimum Remaining Useful Life (RUL) for test windows (used by `mpso`). | `0.0` |
| `--optimizer` | Optimizer used for the `mpso` explainer (`pso`, `da`, `sa`). | `pso` |

### Command Examples

#### 1. CESA Explainer
Run the CESA explainer on the HNEI dataset:
```bash
python test_counterfactual.py --explainer cesa --dataset HNEI --model-path HNEI_CoxPH_model.pkl
```
*This command generates an output file named `exp_results_CESA_HNEI_HNEI_CoxPH.json` by default.*

#### 2. Running a Quick Test
Evaluate on just 2 test windows to verify the pipeline:
```bash
python test_counterfactual.py --explainer cesa --dataset HNEI --max-test-windows 2
```

#### 3. MPSO Explainer
Evaluate MPSO using Particle Swarm Optimization backend:
```bash
python test_counterfactual.py --explainer mpso --dataset HNEI --model-path HNEI_CoxPH_model.pkl --optimizer pso
```
*This command generates an output file named `exp_results_MPSO_HNEI_HNEI_CoxPH.json` by default.*

#### 4. Baseline Explainers (PSO and NN)
Run Particle Swarm Optimization counterfactual search baseline:
```bash
python test_counterfactual.py --explainer pso --dataset HNEI
```

Run Nearest Neighbors counterfactual search baseline:
```bash
python test_counterfactual.py --explainer nn --dataset HNEI
```

---

## 📊 Analyzing Results (`analyze_results.py`)

The `analyze_results.py` script aggregates output JSONs and prints summary statistics (success rate, sparsity, proximity, correlation preservation, and execution times) to the console. It also supports exporting a combined summary table to CSV.

### Command-line Arguments

| Argument | Description |
| :--- | :--- |
| `files` | One or more JSON result files to analyze (positional arguments). |
| `--dir` | Path to a directory containing JSON result files to analyze. |
| `--csv` | Export combined statistics to a CSV file (specifies the output filename prefix). |

### Command Examples

#### 1. Analyze a Single Result File
```bash
python analyze_results.py exp_results_CESA_DO_S_NF_HNEI_CoxPH.json
```

#### 2. Analyze Multiple Result Files (Comparisons)
When multiple files are passed, the script prints summary tables for each, followed by a comparative summary table across all files:
```bash
python analyze_results.py exp_results_CESA_DO_S_NF_HNEI_CoxPH.json exp_results_PSO_HNEI_CoxPH.json exp_results_NN_HNEI_CoxPH.json
```

#### 3. Analyze All JSON Files in a Directory
```bash
python analyze_results.py --dir results/
```

#### 4. Export Combined Summary to CSV
Generate a CSV containing both overall and parameter-specific statistics for paper-ready reporting:
```bash
python analyze_results.py --csv paper_summary exp_results_CESA_DO_S_NF_HNEI_CoxPH.json exp_results_PSO_HNEI_CoxPH.json
```
*(This creates a `paper_summary.csv` file in the current directory).*

---

## 📈 Plotting Results (`plotscode/`)

The repository includes a dedicated `plotscode/` folder with scripts to generate visualizations for the experiments:
- `plot_cesa_models.py`: Plots comparative metrics between different models and datasets.
- `plot_cesa_radar.py`: Generates radar charts comparing different explainers across multiple metrics.
- `plot_cesa_times.py`: Visualizes computation times (e.g. generation vs inference times).
- `plot_delta_effect_grid.py` / `plot_rho_effect_grid.py`: Grid plots illustrating the impact of varying hyperparameters $\delta_1$ and $\rho$ on explanation metrics.
- `analyze_results.py`: (Also linked here) parses and aggregates result JSON files.

You can run these scripts to recreate paper figures and analyze results visually.
