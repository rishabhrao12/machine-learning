# machine-learning

This is a repository for a complete guide on machine learning encompassing both the basics of machine learning as well as deep learning.

## Table of Contents

| Sr. No | Chapter | Models Covered | Projects / Assignments |
| --- | ----------- | ----------- | ----------- |
| **Part 1**|  **Machine Learning** | | |
| 1 | Introduction | - | - |
| 2 | Basics | - | - |
| 3 | Classification | SGDClassifier, SVC, KNN | MNIST, Titanic, Spam Classifier |
| 4 | Regression | LinearRegression, SGDRegressor, RidgeRegression, LassoRegression, ElasticNet, LogisticRegression | Real Estate Prices, Medical Insurance |
| 5 | SVM | LinearSVC, SVC, LinearSVR, SVR | MNIST, California Housing |
| 6 | Decision Trees | DecisionTreeClassifier, DecisionTreeRegressor | Moons, Random Forest |
| 7 | Ensemble Learning | | |
| 8 | Dimensionality Reduction | | |
| 9 | Unsupervised Learning | | |
| 10 | Forecasting | ARIMA, SARIMA, SARIMAX, Prophet| |
| 11 | Reinforcement Learning | | |
| **Part 2**|  **Deep Learning** | |
| 12 | Introduction | |
| 13 | Deep Neural Networks | | |
| 14 | Proccessing Data with Tensorflow | | |
| 15 | Deep Computer Vision | CNN | |
| 16 | Processing Sequences | CNN, RNN | |
| 17  | Natural Language Processing | RNN | |
| 18 | Representation and Generative Learning | GAN | |
| 19 | Reinforcement Learning | | |
| 20 | Deploying Models at Scale| | |
| 21 | Forecasting | CNN, RNN, LSTM, N-BEATS | |

## How to make use of this repository?

This repository is divided into two parts: Machine Learning and Deep Learning. The structure of both these parts are the same and is explained as follows: 

The parts themselves are divided into subsequent chapters. Each chapter has a directory with the same name. Each chapter will have one notebook, which is named as the chapter, this is the notebook where you should start reading up on the topic. Check the table of contents to see what the chapter is, the models covered and the assignments or projects that will be assigned. The notebook covers all the theory for the topic and also has the code which you can run to see the models in action. Every one of these topic wise notebooks has an exercises section at the end. Exercises can contain either assignments or projects, all assignment files are prefixed with A1, A2... and the projects are preffixed by P1, P2... The projects are larger scale, either more complex or they work on large datasets or both. Projects should be attempted after you complete the assignments. Feel free to come back to a project later as well, as you might learn something in subsequent sections that could help you improve your project. 

The directory will also contain files on concepts, for example in regression we have implementations on the various GradientDescent topic. These concept files can help you understand how various algorithms work under the hood, they do not use any of the built in sklearn functions. The concept files are named as the concept they cover, for example, GradientDescent.ipynb. The concept files are not mandatory to read, but they are highly recommended. The concept files are also not in any particular order, you can read them in any order you want. The concept files are also not exhaustive, they are just meant to give you a basic idea of how the algorithm works. If you want to learn more about a particular algorithm, you can search for it on the internet, there are plenty of resources available. These concepts are also covered in the theory but the concept files contain the code to implement them without having to make use of any prebuilt functions from sklearn or tensorflow.