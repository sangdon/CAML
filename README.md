# CAML (Confidence Auditor-based on Manifold Learning)

CAML predicts confidence of a label prediction by a given classifier. Internally, CAML model a calibration predictor by using manifold learning. 

## Prerequisites
* Linux
* Python 3.5+
* Pytorch 1.0+

## Getting Started

### Installation
First, clone this repository. 
```
git clone https://github.com/sangdon/CAML.git
```

### Dataset Initialization
For training and evaluation, initialize datasets. Let the root of a dataset be `<dataset root>`. Under the folder `<dataset root>`, CAML requires three folders `<train>`, `<val>`, `<test>` for training, hyperparameter tuning, and evaluation for CAML, respectively. Under the three folders, folders with label names are located, `<label 1>`, `<label 2>`, ...,  `<label K>`, where K is the number of labels. Each label folder `<label i>` contains images in png file format. Note that this folder structure follows the convention of pytorch `ImageFolder`, such that CAML can use `ImageFolder` to generally read any datasets. 

#### Example
For example, the following script initialize the MNIST dataset at `user_input/datasets/MNIST` with the above mentioned folder structure. 
```
cd user_input/datasets
python3 init_MNIST_dataset.py
```

## Execution Overview

The CAML can be used in three phases, 1) training, where CAML trains its internal representations, 2) evaluation, where the calibration error of CAML is evaluated on a test set, and 3) running, where a trained CAML model reads one image and predict a confidence. For each phase, the high-level description of the input/output of CAML is as follows:

* training: `<classifier>`, `<dataset>`, `<base options>`, `<training options>` --> `CAML` --> `<CAML model>`
<p align="center">
<img align="center" src="https://github.com/sangdon/CAML/blob/master/doc/CAML_training.png" width=500>
</p>

* evaluation: `<classifier>`, `<dataset>`, `<base options>`, `<evalutation options>`, `<CAML model>` --> `CAML` --> `<calibration error>` 
<p align="center">
<img align="center" src="https://github.com/sangdon/CAML/blob/master/doc/CAML_eval.png" width=500>
</p>

* running: `<classifier>`, `<dataset>`, `<base options>`, `<running options>`, `<CAML model>` --> `CAML` --> `<confidnece>`
<p align="center">
<img align="center" src="https://github.com/sangdon/CAML/blob/master/doc/CAML_running.png" width=500>
</p>

Here, each input, `<classifier>`, `<dataset>`, `<base options>`, `<training options>`, `<evalutation options>`, `<running options>` are specified by options of CAML. For outputs, `<CAML model>` represents a folder where CAML model is saved, `<calibration error>` and `<confidnece>` are scalar numbers printed in a screen.

### Description on Options

`<classifier>` means a target classifier, where it consists of a pytorch network definition `<net>.py`, a class name `<net class name>`, and network parameters `<parameters file name>.pt`. Each `<net>`, `<net clss name>`, and `<parameters file name>` is set by options of CAML, `--model_def_path=<net>.py`, `--model_name=<net class name>`, and `--model_path=<parameters>.pt`.

`<dataset>` means a target dataset, where it consists of the location of dataset root `<dataset root>`, the size of image `<image size>`, a flag for gray scale `<use gray scale>`. Each information is specified as CAML options as follows: `--dataset_root=<dataset root>`, `--image_size=<image size>`, and `--gray_scale=<use gral scale>`.

`<base options>` includes basic options `<exp name>`, `<use gpu>` and `<n_manifolds>`, required anytime. Here, `<exp name>` is an experiment name specified in `--exp_name=<exp name>`, `<use gpu>` is the flag whether use GPU specified in `--use gpu=<use gpu>`, and `<n_manifolds>` is the number of manifolds to generate by CAML in `--n_manifolds=<n_manifolds>`. 

`<training options>` contains required hyperparameters for CAML, which includes batch size `<batch size>` to read images in `--batch_size=<batch size>`, a retrain option `<relearn model>` in `--relearn_model=<relearn model>`, a lower bound for hyperparameter line search `<lb>` in `--width_max_lb=<lb>`, an upper bound for hyperparameter line search `<lb>` in `--width_max_ub=<ub>`, a step size for hyperparameter line search `<step>` in `--width_search_delta=<step>`, a maximum roation angle in degree `<angle_max>` to generate manifolds in `--rotation_max=<angle_max>`, and a rotation incremental `<angle_delta>` in `--rotation_delta=<angle_delta>`. 

`<evalutation options>` contains batch size `<batch size>` to read images in `--batch_size=<batch size>`, and the number of bins `<n_bin>` to approximately measure a calibration error in `--n_bins=<n_bin>`. `<running options>` contains the path `<image path>` of input image in `--image_path=<image path>`.

The usage of CAML in each phase is described in the following sections.


## Train CAML
To train a CAML model, execute `train_CAML.py` with proper options. To see the possible options, check `python3 train_CAML.py --help`.

### Example
The following comments train CAML on a MNIST dataset with a trained LeNet5 classifier:
```
python3 train_CAML.py @user_input/opts_train.txt
```

## Evaluate CAML
To evaluate a CAML model on a test set, execute `eval_CAML.py` with proper options. To see the possible options, check `python3 eval_CAML.py --help`.

### Example
The following comments evaluate CAML on a MNIST test set with a trained LeNet5 classifier:
```
python3 eval_CAML.py @user_input/opts_eval.txt
```

## Run CAML
To run a trained CAML model, execute `run_CAML.py` with proper options. 

### Example
The following comments predict a confidence of a trained LeNet5 classifier using CAML on a given image:
```
python3 run_CAML.py --image_path examples/example.png @user_input/opts_run.txt
```
