LapImpute: Graph Node Feature Imputation
========================================

This repository contains the code for LapImpute, a graph-based node
feature imputation method built on top of GNNs and k-NN style
initialization. The implementation focuses on imputing missing node
features on citation, web, and heterophilous graphs under various
missing-rate settings.

Directory layout
----------------

Key files and folders:

- `lapimpute.py`: main training / imputation script for LapImpute.
- `lapimpute.sh`: shell script to run LapImpute across multiple datasets.
- `util_funcs.py`: utility functions for dataset loading, k-NN
  imputation, similarity computation, evaluation metrics, and plotting.
- `data/`: root directory for graph datasets (created by you).
- `logs/`: log files from LapImpute runs.

Environment
-----------

The code was developed and tested with the following library versions:

- Python: **3.9.22**
- PyTorch (`torch`): **2.0.1+cu117**
- PyTorch Geometric (`torch_geometric`): **2.6.1**

Other Python dependencies:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `hnswlib`
- `torch_sparse`
- `torch_scatter`
- `torch_geometric` datasets and utilities

Example conda setup (adapt the environment name, versions, and CUDA build
to your system):

```bash
conda create -n lapimpute python=3.9
conda activate lapimpute

# Install PyTorch (choose the correct CUDA build for your GPU/driver)
pip install torch==2.0.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install PyG and related packages; see https://pytorch-geometric.readthedocs.io
pip install torch-geometric==2.6.1 torch-sparse torch-scatter

# Other dependencies
pip install numpy scipy scikit-learn matplotlib hnswlib
```

Data
----

LapImpute expects datasets to live under the local `data/` directory
in the usual PyTorch Geometric layout. Examples include:

- Planetoid-style citation networks (e.g., Cora, Citeseer, Pubmed)
- WebKB and related Web graphs (e.g., Cornell, Texas, Wisconsin)
- Amazon / Coauthor graphs
- HeterophilousGraphDataset benchmarks under `data/hetero`

Most datasets are downloaded automatically by `torch_geometric` the
first time they are requested, using the root paths in `util_funcs.py`
(`./data` and `./data/hetero`).

Running LapImpute
-----------------

From this directory, after activating your Python environment:

```bash
bash lapimpute.sh
```

The script `lapimpute.sh` iterates over a predefined list of datasets and
calls `lapimpute.py` with default hyperparameters (missing rate, group size,
training epochs, etc.). To run a single dataset or override arguments directly,
call the Python entrypoint:

```bash
python -u lapimpute.py --dataset Cora --mr 0.6 --group_size 10
```

During training, logs are written into `logs/`, and intermediate
artifacts such as filled features and sparse adjacency matrices are
stored in `features/` and `data/` subdirectories created on demand.

Reproducibility tips
--------------------

- Keep a copy of the exact `conda list` or `pip freeze` from your
  environment for future reference.
- When changing GPU drivers or CUDA versions, ensure PyTorch and PyG are
  reinstalled with matching CUDA builds.
- Use the same dataset versions and splits as produced by
  `torch_geometric` to avoid subtle distribution differences.
