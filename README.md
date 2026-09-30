# 2026-human-ibl-analyses
Model-free analyses of behavior of the human IBL dataset.

## Installation
Create the `human-ibl` conda environment (Python 3.12 and all dependencies) with:

```
conda env create -f environment.yml
conda activate human-ibl
```

Alternatively, with an existing Python 3.12+ installation:

```
pip install -r requirements.txt
```
## Download data
The data used to run the analyses can be found at [ZENODO]. We suggest to download them and save them in a data/ folder.
If you change its name or path, make ure you change the paths also in the scripts

## Reproducing the figures
The scripts in the folder `src` can be run to reproduce the figures. The figures are then saved into the folder `figures`.

## Transforming from raw data
Raw data can be made available upon request to the authors. The preprocessing functions to extract and transform the raw data into the published form are provided in the folder `preprocessing`. 
