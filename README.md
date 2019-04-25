# CAML (Confidence Auditor-based on Manifold Learning)

CAML predicts confidence of a label prediction by a given classifier. Internally, CAML model a calibration predictor by using manifold learing. 

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
For training and test, initialize datasets. For example, the following script initialize the MNIST dataset. 
```
cd user_input/datasets
python3 init_MNIST_dataset.py
```
The folder structure of initialized dataset is `<dataet name>/<train/val/test>/<label name>/<image files>` which follows the convention of pytorch `ImageFolder`. 

### Train CAML
To train a CAML model, execute `train_CAML.py` with proper options. The option example is provided at `user_input/opts.txt`.
```
python3 train_CAML.py @user_input/opts.txt
```

### Run CAML
To run a trained CAML model, execute `run_CAML.py` with proper options. In running, providing image path to test is required as follows:
```
python3 run_CAML.py --image_path examples/example.png @user_input/opts.txt
```
