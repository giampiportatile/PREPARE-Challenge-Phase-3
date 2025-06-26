# PREPARE-Challenge-Phase-3
Proof of Principle Demonstration for Phase 3 of PREPARE CHALLENGE  "Pioneering Research for Early Prediction of Alzheimer's and Related Dementias EUREKA Challenge" .

Link to the phase 3 of the competition: https://www.drivendata.org/competitions/304/prepare-challenge-phase-3/

The project structure is based on https://cookiecutter-data-science.drivendata.org/.

Author: Gianpaolo Tomasi

Team: GiaPaoDawei

Licence: MIT

This folder contains the trained models, that can be directly used to generate the forecast for the test set.

For TabM we saved 5 different versions corresponding to 5 different seeds (as neural networks are not deterministic).

We could not save the fitted TabPFN model is it is too big for gitHub (>100Mb).

The code used to fit all these models is saved in ./src/train_models.py
