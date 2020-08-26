# Independant Research Project
## Exploring regression with WGANs

Code produced for the paper "Exploring regression with WGANs" by Ellyess Benmoufok, supervised by: Christopher Pain, Alexandra Porter and Toby Phillips.

## Getting Started

### Dependencies
All the `Python` library prerequistes can be found and installed via `requirements.txt`:
* jupyter
* numpy
* pandas
* sklearn
* tensorflow, keras
* GPy https://sheffieldml.github.io/GPy/

### Installation & Usage:
* Clone the repository from Github by either:
  * Using command line:
  `git clone https://github.com/acse-2019/irp-acse-efb119.git`
  * Downloading the repository as a .zip
* Installing all the requirements by using command line:
 `pip install -r requirement.txt`
* Run the notebooks using `jupyter-notebook`

### Notebooks:
* Run `run_models.ipynb` for synthetic datasets
* Run `run_models_silverdata.ipynb` for silver nanoparticle dataset

### The Data
* Synthetic datasets found in `datasets.py`: `sinus`, `circle`, `multi`, `3d`, `moons` taken from `sklelarn.datasets`, `helix, `eye` and `heter`
* Any datasets requiring files can be found in the `data` folder.

### Models
* All functions (preprocessing, training, etc) associated with the WGAN-GP can be found in `WGAN_model.py`, the WGAN network configuration can be found in `network.py`.
* All functions associated with the GPR can be found in `GPR_model.py`.


## Further information:

This code is directly associated with a project report, more information on motivation and purpose can be found in that paper. Code is commented for understanding of functions.
