# Gaussian Naive Bayes Implementation


## Introduction

This module presents 2 different implementations of the same Naive Bayes Classifier:
* Using machine learning framework scikit-learn.
* From scratch using pandas library.

The implemented model is a basic Naive Bayes Classifier with the following attributes:
* Continuous variables.
* Gaussian naive Bayes assuming the continuous values are distributed according to a Gaussian distribution.
* Independant features (Naive hypothese)
* Binomial classifier


## Motivation

The purpose of this exercice is to gain a better understanding of Naive Bayes classifier, as it is one of the simplest yet still one of the best choice in many practical applications.

It is also a valuable exercice to practice and improve python programming skills.

## Code structure

The code is structured as follow:
```
pyNeuralNet
├- data
│   ├- titanic
│   │   ├- raw_data
│   │   │   └- data.csv
│   │   └- data.pkl
│   └- us_election
│       ├- raw_data
│       │   └- data.csv
│       └- data.pkl
├- library
│   ├- doityourself
│   │   ├- params
│   │   └- model.py
│   ├- random
│   │   ├- params
│   │   └- model.py
│   └- scikit-learn
│       ├- params
│       └- model.py
├- performance
│   ├- qualitative.py
│   └- quantitative.py
├- .gitignore
├- evaluate.py
├- predict.py
├- README.md
├- requirements.txt
├- tools.py
└- train.py
```

The module has 3 main fucntionnalities:
* train.py: fit a model given a set of data.
* predict.py: perform a prediction given a previoulsy fitted model.
* evaluate.py: evaluate the performance a model using different metrics.


## Installation

To dowload the different implementations of neural networks, you can directly clone the repository
```
git clone https://github.com/lamsremi/pyNaiveBayes.git
```
Then install the requirements in your environment or in a virtual one
```
pip install -r requirements.txt
```

## Run

```
# Training
>>> from train import main
>>> for source in ["us_election", "titanic"]:
        for model in ["random", "doityourself", "scikit_learn]:
            main(data_df=None,
                 data_source=source,
                 model_type=model)

# Prediction
>>>from predict import main
>>> X_INPUT = pd.Series({"popul": 300, "TVnews": 3, "selfLR": 3, "ClinLR": 3,
        "DoleLR": 5, "PID": 1, "age": 45, "educ": 4, "income": 15 })
    for source in ["us_election"]:
        for model in ["doityourself", "random", "scikit_learn"]:
            main(x_input=X_INPUT,
                 model_type=model,
                 model_version=source)
```

## Author

Rémi Moise
moise.remi@gmail.com

## Licence

MIT License

Copyright (c) 2018 Rémi Moïse
