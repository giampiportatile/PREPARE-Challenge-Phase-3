# PREPARE-Challenge-Phase-3
Proof of Principle Demonstration for Phase 3 of PREPARE CHALLENGE  "Pioneering Research for Early Prediction of Alzheimer's and Related Dementias EUREKA Challenge" .

Link to the competition: https://www.drivendata.org/competitions/group/nih-nia-alzheimers-adrd-competition/

Link to the phase 3 of the competition: https://www.drivendata.org/competitions/304/prepare-challenge-phase-3/

The project structure is based on https://cookiecutter-data-science.drivendata.org/.

Author: Gianpaolo Tomasi

Team: GiaPaoDawei

Licence: MIT


## Summary
The challenge is centered around developing better methods for prediction of Alzheimer's disease and Alzheimer's disease related dementias (AD/ADRD) focuing in particular on early diagnosis. 
Phase 3 is focused on the imporvement of the winning models of Phase 2 for both social determinant and acoustic tracks.


## Setup

Using pip:
1) create virtual environment
conda create -n PREPARE_phase3 python=3.9

2) Install the required libraries from the requirements.txt file:

pip install -r requirements.txt


Using conda:

Create a new environment directly from the .yaml file shared in this directory:

conda env create -n PREPARE_phase3  -f environments.yaml


## Data
Copy the 5 files of the competition (social determinats track) into the data/raw folder.
Note the raw data is not available anymore on the DrivenData website, so we will assume the user downloaded them during the competition.


train_features.csv
train_labels.csv
test_features.csv
submission_format.csv


To skip training, copy the model weights to models/model.pkl

To use the the Jupyter notebook examples.ipynb with this environment:

pip install --user ipykernel
python -m ipykernel install --user --name=nr_prepare
And select nr_prepare as your Python kernel in Jupyter.


Hardware

Training time: ~3m 10s
Inference time: ~3s
Training and inference were both run on CPU.

Run training
The model can be trained from the command line or in Python.

Command line training
To run training from the command line: python src/train.py.

$ python src/train.py --help
Usage: train.py [OPTIONS]

Options 
  --features-path           PATH    Path to the raw training dataset for processing
                                    [default: data/raw/train_features.csv]
  --labels-path             PATH    Path to the training labels
                                    [default: data/raw/train_labels.csv]
  --model-save-path         PATH    Path to save the trained model weights
                                    [default: models/model.pkl]
  --cv                              Cross validate on training dataset and report RMSE before training
                                    [default: no-cv]                                       
  --cv-predict                      Generate predictions from cross validation and save before training
                                    [default: no-cv-predict]
  --cv-predictions-path     PATH    Path to save predictions from cross validation
                                    [default: data/processed/cv_predictions.csv]
  --debug                           Run on a small subset of the data for debugging
                                    [default: no-debug]
 --help                             Show this message and exit.
