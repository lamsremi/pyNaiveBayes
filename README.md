# Implementation of Gaussian Naive Bayes model


## Presentation

P(C_k|x): probability of having each of the class C_k given the input x (posterior)
P(C_k|x) = P(x|C_k) * P(C_k) / P(x): Bayes theorem
    - P(x|C_k): probability of having the input x given the class C_k (likelihood)
    - P(C_k): probability of having the class C_k (prior)
    - P(x): probability of having the input x (evidence)


## Pseudocode

```
# Prediction
predicted_y = argmax_k(P(C_k|x)) [decision rule]
    P(C_k|x) = P(x|C_k) * P(C_k)
        P(C_k) = equiprobable classes or probability classes using occurences
        P(x|C_k) = II_i P(x_i|C_k) (because independant variable)
            P(x_i|C_k) = Gaussian for continuous or multinomial and bernouilli for categorical


# Training
For Gaussian
For each of the class
    For each of the continuous variable
        Compute the gaussian parameters
    Compute the probability of the class
```

