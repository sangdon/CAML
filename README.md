# CAML (Confidence Auditor-based on Manifold Learning)

## Prerequisites
* Linux
* Python 3.5+
* Pytorch 1.0+

## Getting Started

### Installation
```
git clone https://github.com/sangdon/CAML.git
```

### Dataset Initialization

```
cd user_input/datasets
python3 init_MNIST_dataset.py
```

### Train CAML

```
python3 train_CAML.py @user_input/opts.txt
```

### Run CAML
```
python3 run_CAML.py --image_path examples/example.png @user_input/opts.txt
```
