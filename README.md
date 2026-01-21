# CropClimateX
Repository for the paper: "A large-scale, multitask, multisensory dataset for climate-aware crop monitoring in the United States from 2018–2022" by Adrian Höhl, Stella Ofori-Ampofo, Miguel-Ángel Fernández-Torres, Ridvan Salih Kuzu, and Xiao Xiang Zhu. More information about the dataset can be found on [Hugging Face](https://doi.org/10.57967/hf/5047).
## Usage
### Download
Dataset can be download from Hugging Face:
- website (not recommended for large-scale download): `https://huggingface.co/datasets/torchgeo/CropClimateX`
- via git:
`git clone https://huggingface.co/datasets/torchgeo/CropClimateX`
- or download with a script from Hugging Face API:

    ```python src/datasets/download.py```

    ```python src/datasets/download.py -m <modalities> -ids <geoids> --local_dir <download_folder> -nr_workers <number parallel downloads>```
### Folder structure
Each data source is located in a folder, each folder contains the minicube .zarr files, one .zarr file contains up to 10 minicubes. Each minicubes has an id like this: \<GEOID\>_\<PID\>, where the GEOID is the id of the county and PID is the id of the minicube.
### File format
The xarray library is recommend to read the files: `xr.open_zarr(filename, group=minicube_id)`. The data is saved as integers to save space, xarray applies the scaling and offset automatically to the data while loading it. Also, all meta data like coordinates and time are loaded correctly.

You can preprocess and change the format of the files into .npy/.zarr by using the script in datasets/dataset_format.py. All the resampling/selection will be taken care of by reusing the dataloader and saving it to new files.
## Dataset creation
### Minicube Sampling
The location of the minicubes were optimzed for each county. Two alogrithms were used a Genetic Algorithm and a Sliding Grid Algorithm, these were compared to a straightforward gridding of the county (baseline). They can be run using the following scripts (see also [install environment](#Installing-the-environment)):
```
python src/dataset_preparation/minicube_creation/main_gp.py
python src/dataset_preparation/minicube_creation/main_sga_baseline.py
```
## Running experiments
This repository includes example code demonstrating how to run deep learning experiments using the provided dataset. The repository is based on Pytorch Lightning and Hydra.
In the files src/datasets/minicube_dataset.py src/datasets/yield_dataset.py you can find PyTorch code to load the data in minicube or county/yield format.
### Installing the environment
Run this to install the dependencies (you may need a different pytorch version for your hardware etc.):
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
or install with uv and exact python version:
```
pip install uv # install package manager
uv venv --python 3.10.13
source .venv/bin/activate
uv pip install -r requirements.txt
```
### Doing Experiments for Crop Yield
A few sample architectures can be found in configs/experiments/ and src/models/.
download data:
```
python src/datasets/download.py -m cdl_500m modis daymet dem soil yield --local_dir data/CropClimateX
```
preprocess/resample data (if needed):
```
python src/datasets/dataset_format.py --ds_name=prep_yield_modis_corn_float32 --data=yield_modis_data_prep --bands sur_refl_b01 sur_refl_b02 sur_refl_b03 sur_refl_b04 sur_refl_b06 tmax tmin prcp vp srad elevation slope aspect curvature bulk_density cec clay ph sand silt soc --pred_bands yield --bands_channel_dim=-3 --dtype=float32
```
example for training:
```
python src/main.py train=True test=True seed=42 experiment=final_yield_cnn_lstm
```
example for testing with weights:
```
python src/main.py train=False test=True experiment=final_yield_cnn_lstm ckpt_path=weights/yield/cnn_lstm_seed\=42.ckpt 
```
### Developing new models
1. clone/fork the repo
2. implement your model in src/models/
3. create/modify a config file configs/models (& configs/data)
4. train the model with `python src/main.py ...`
#### Running scripts with Hydra
- hyperparameter tuning:
`python src/main.py --multirun logger=wandb model.optimizer.lr=0.1,00.1 tags=[tune_lr]`
- hyperparameter tuning with optuna:
`python src/main.py -m hparams_search=<config>`
- cross validation:
`python src/main.py --multirun data.split.nr_folds=6 data.split.k=0,1,2,3,4,5 tags=[<tag>]`
- for more see: https://hydra.cc/docs/intro/
## Citation
If you use this work or the dataset please consider citing:
```
@article{hohlLargescaleMultitask2026,
  title = {A Large-Scale, Multitask, Multisensory Dataset for Climate-Aware Crop Monitoring in the {{US}} from 2018--2022},
  author = {H{\"o}hl, Adrian and {Ofori-Ampofo}, Stella and {Fern{\'a}ndez-Torres}, Miguel-{\'A}ngel and Kuzu, R{\i}dvan Salih and Zhu, Xiao Xiang},
  year = 2026,
  month = jan,
  journal = {Scientific Data},
  issn = {2052-4463},
  doi = {10.1038/s41597-026-06611-x}
}
```
## License
The code is licensed under the MIT license.
## Acknowledgements
Code Template: https://github.com/ashleve/lightning-hydra-template