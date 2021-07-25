# Multitask Learning for Garment Attributes Classification

This is a PyTorch implementation for classifying attributes of garments, such as neck_type, sleeve_length, and pattern.

It uses Resnet till the second last BasicBlock to predict the global features. These global features are fed into three separate task heads which predict the categories of each task.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
.
+-- checkpoints
|   +-- 1 (denotes the experiment seed)
|   |   +-- best.pth
|   |   +-- current.pth
+-- data
|   +-- attributes.csv   (original csv file)
|   +-- attributes_clean.csv   (after cleaning the dataframe)
|   +-- images  (original image folder)
|   +-- sample_data   (image folder to run inference script)
+-- utils
|   +-- __init__.py
|   +-- model_builder.py   (creates user defined resnet)
|   +-- resnet.py   (resnet main code)
|   +-- utils.py   (utils for loading pretrained resnet from url)
+-- __init__.py
+-- args.py   (contains the configuration of the project)
+-- basic_eda.ipynb
+-- dataset.py   (dataloader)
+-- inference.py   (inference script)
+-- main.py   (main file that runs the training and testing)
+-- model.py   (defines the multi-task learning model)
+-- requirements.txt
+-- trainer.py   (contains the trainer class)

```


## Usage

Generate attributes_clean.csv from basic_eda.ipynb file.

To train the network:
```bash
python main.py --seed 1 --phase "train"
```
To resume training of a particular experiment (say 5):
```bash
python main.py --seed 5 --phase "train" --resume_training True
```
To see the performance on the test set, run:
```
python main.py --seed 5 --phase "test"
```
Similarly, there are many parameters (like loading pre-trained resnet, using gpu, changing the resnet arch etc.) that can be configured in args.py, After changing those parameters just run:
```
python main.py
```

To run the inference on a given folder apart from the experiment test set. We need to run inference.py
```
python inference.py --infer_dir "./data/sample_data"
```
it will generate output.csv file
