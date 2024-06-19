# Positive and Unlabeled Multi-Instance Support Vector Machine (PUMISVM)
This repository contains a collection of Python scripts implementing the Positive and Unlabeled Multi-Instance Support Vector Machine (PUMISVM) model and its experiments on synthetic and Chest X-Ray PU multi-instance datasets. It covers data creation, building baseline models, reporting classification performance metrics, and visualizing the results.

## Project Description

- `run.py`: Imports baseline models from [kenchi](https://github.com/Y-oHr-N/kenchi), [pyod](https://github.com/yzhao062/pyod), [sklearn](https://scikit-learn.org/stable/), and our PUMISVM model. Configures parameters, creates the Positive and Unlabeled Multi-Instance dataset, splits the dataset, conducts PU learning and classification, and reports results in terms of accuracy, precision, recall, F1 score, and balanced accuracy.
- `utils.py`: Contains utility functions for experiments, processes raw chest X-ray data, creates the multi-instance dataset for PU learning, and includes visualization code for plotting important instances identified by the PU learning model.
- `PUMISVM.py`: Implements the PUMISVM model, featuring the class PUMISVM with three key methods: __init__, fit, and predict. The __init__ method defines the hyperparameters of PUMISVM. The fit method takes the multi-instance training data X and its associated grouping information y (with ungrouped bags marked as 'unlabeled'). The predict method also takes multi-instance data X and their grouping information, outputting decisions where 1 indicates an inlier and -1 indicates an outlier for the bags in X.
- `other_models.py`: Includes baseline Positive and Unlabeled (PU) learning or outlier detection models for comparison. Contains wrappers for sklearn, kenchi, and pyod models, with each wrapper defining the fit and predict methods to consistently output 1 for inliers and -1 for outliers from multi-instance data.

## Getting Started
### Dependencies

- Python 3.9.16
- Python packages listed in requirements.txt

### Installation
To set up the project, start by cloning the repository to your local machine. It is highly recommended to create an isolated Python environment via conda. For example:
```
conda create -n "PUMISVM" python=3.9
```

Then, activate the created environment:
```
conda activate "PUMISVM"
```

Next, install the required packages:
```
pip install requirements.py
```

## Running the tests

After installing the packages, you can conduct the experiments by running:

```
python3 run.py
```
A folder will be created under the 'output_path' in `run.py`, containing all the experimental results.

## Data

The necessary data for executing the framework's scripts is readily accessible in the [Chest X-ray dataset repository](https://github.com/ieee8023/covid-chestxray-dataset). Clone this repository to your local machine and specify the path to the cloned dataset in `run.py`.

## Contributing

We welcome contributions to improve the framework and extend its functionalities. Please feel free to fork the repository, make your changes, and submit a pull request.
