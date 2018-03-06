# Gaussian Naive Bayes


## Introduction

This module presents 2 different implementations of a Naive Bayes Classifier:
* In pure python.
* And using scikit-learn framework.


## Motivation

The purpose of this exercice is to gain a better understanding of Naive Bayes classifier, as it is one of the simplest yet still one of the favorite choice in many practical applications.

It is also a valuable exercice to practice python programming.


## Information gained

# Gaussian Naive Bayes

Gaussian Naive Bayes is part of the family of Naive Bayes classifiers, which apply Bayes theorem.

The application of the theorem to compute conditional probability is made possible with the assumption that all the features are independent between each others, allowing to compute the probabilties for each feature separately and combine them together.

To model the probability distribution of a value given the a class, Gaussian Naive Bayes assume that the continuous values associated with each class are distributed according to a Gaussian distribution.

Despite oversimplified assumptions, GNB have works well in many real world application due to his ease of training.

# Python programming

## Code structure

The code is structured as follow:
```
pyNaiveBayes
├- data/
│   │
│   ├- titanic/
│   │   ├- raw_data
│   │   │   └- data.csv
│   │   ├- data.pkl
│   │   └- preprocess.py
│   │
│   └- us_election/
│       ├- raw_data
│       │   └- data.csv
│       ├- data.pkl
│       └- preprocess.py
│
├- library/
│   │
│   ├- pure_python/
│   │   ├- params
│   │   └- model.py
│   │
│   └- scikit-learn/
│       ├- params
│       └- model.py
│
├- benchmark/
│   └- performance.py
│
├- evaluate.py
├- predict.py
├- prepare.py
├- train.py
│
├- README.md
├- .gitignore
└- requirements.txt
```

The module has 3 main fucntionnalities:
* train.py: fit a model given a set of data.
* predict.py: perform a prediction.
* evaluate.py: evaluate the performance a model.


## Installation

To clone the repository :
```
git clone https://github.com/lamsremi/pyNaiveBayes.git
```

To install the dependencies :
```
pip install -r requirements.txt
```

## Run

```
# Training
>>> import train
>>> train.main(data_source="us_election",
               model="pure_python")

# Prediction
>>> inputs_data = [row[0] for row in load_labaled_data("us_election")][0:100]
>>> for model in ["pure_python", "scikit_learn"]:
        predict.main(inputs_data=inputs_data,
                     model_type=model,
                     model_version="X")
```

## Author

Rémi Moise
moise.remi@gmail.com

## Licence

MIT License

Copyright (c) 2018 Rémi Moïse
