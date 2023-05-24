# CL-UZH-EDOS-2023

This repository contains the code to reproduce the results of the paper _CL-UZH@EDOS2023: Incremental Fine-Tuning and Multi-Task Learning with Label Descriptions_.

## Run Experiments

Setup the environment:
```bash
python -m venv env
source env/bin/activate.bin
pip3 install -r requirements.txt
```

Download, split and preprocess the datasets:

```bash
cd CL-UZH-EDOS-2023
bash download.sh
bash preprocess.sh
```

Execute the experiments:

```bash
bash execute_experiments.sh
```
