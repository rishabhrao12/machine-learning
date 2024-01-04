# Introduction to Machine Learning

This chapter and the resources within it are to provide an introduction to machine learning as well as a brief idea about the landscape in machine learning.

The topics to be covered in this section are as follows: -
1. What is machine learning?
2. Why use machine learning?
3. Application of Machine Learning
4. Types of Machine Learning Systems
5. Main Challenges of Machine Learning
6. Testing and Validating
7. Exercises

Since this is mainly a theory based chapter the only content present will be either links to blog articles or notebooks with theory, pictures and code snippets.


## 1. What is machine learning?

Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.

A computer program is said to learn from experience E with respect to some task T and some performance measure P, its performance on T, as measured by P, improves with experience E.

A spam filter is a Machine Learning program that given examples of spam emails and regular emails can learn to flag spam, The examples that the system uses to learn are called the training set.
Each training example is called a training instance (or sample). In this case task T is to flag spam for new emails, experience E is training data and performance measure P needs to be defined. 
For example, you can use the ratio of correctly identified emails. This is often called accuracy in classification tasks.

If you download a copy of some data that means you have more data it does not mean your computer is better at machine learning.


## 2. Why use machine learning?

Machine learning is great for : -
1. Problems for which existing solutions require a lot of fine-tuning or long lists of rules: one machine learning algorithm can often simplify code and perform better than the traditional approach.
2. Complex problems for which a traditional approach yields no good solution: the best machine learning techniques can perhaps find a solution.
3. Fluctuating environments: a machine learning system can adapt to new data.
4. Getting insights about complex problems and large amounts of data.

E.g. 1) An example of this is if we take our spam filter example, if we wanted to make this using the traditional approach we would have tp perform the following steps: -
1. First, we would consider what spam looks like: you might notice typical words being used like "4U", "free", "credit card". You might notice these words appear a lot in the subject. You might also notice other patterns in the senders mail, the emails body etc.
2. You would write a detection algorithm for each of the patterns you noticed and if a certain amount of patterns is noticed it is flagged as spam.
3. You would test the program and repeat steps 1 and 2 till the algorithm is good enough to be used.

Since the problem is difficult it is likely that the program will become a long list of complex rules which will be hard to maintain, as opposed to an ML solution where the algorithm will learn automatically which words and phrases are common to be found in spam and is hence a good spam detector.
Also, if attackers notice some words like "4U" are getting blocked they might start using "For u" instead and hence we would have to update our algorithm, instead in an ML solution it would automatically learn that "For u" is also being found in spam.

E.g. 2) In a voice recognition program if we are trying to classify words "apple" and "orange", the steps to take this in a traditional approach will be highly complex as we will have a large number of individual rules for each word such as the pitch and there will be a lot of other variables involved. To avoid this complexity an ML solution is much better at learning patters from the data and is not as complex.

We can also use the patterns found machine learning algorithms to learn more about the data, e.g. in the spam filter we could find what words are often used in spam mails.


## 3. Applications of Machine Learning

Some concrete examples of machine learning and techniques that can be used to tackle them are as follows: -

1. Analyzing images of products on a production line ot automatically classify them: this is image classification typically performed using convolutional neural networks.
2. Detecting tumours in brain scans: this is semantic segmentation, where ech pixel in the image is classified (as we want to determine the exact location and shape of the tumor), typically using CNN's.
3. Automatically classifying news articles: this is natural language processing (NLP) and more specifically text classification which can be tackled using recurrent neural networks (RNN) and convolutional neural network (CNN)
4. Summarizing long documents: this is a branch of NLP called text summarization.
5. Detecting card fraud: this is anomaly detection.

## 4. Types of Machine Learning Systems

### 4.1 Types of Machine Learning

There are so many types of machine learning that it makes sense to categorise them into broad categories based on the following criteria: -

1. Whether they are trained with human supervision or not (Supervised, Unsupervised, Semi-supervised, Reinforcement Learning)
2. Whether they can learn incrementally on the fly or not (Online vs Batch Learning)
3. Whether they work by simply comparing new data-points to known data-points or they find patterns in the data and build a predictive model (Instance based Learning vs Model based Learning)

These criteria are not exclusive and instead can be combined. For example a state-of-the-art spam filter may learn on the fly using a deep neural network model trained using examples of spam and ham; this makes it an online, model based, supervised learning system.

Machine Learning systems can be classified according to the amount and type of supervision they get, as a result they can be classified as Supervised, Unsupervised, Semi-supervised and Reinforcement.

#### 4.1.1 Supervised Learning

In supervised learning, during the time of training the training set you feed the algorithm includes the desired solutions called the labels.

A typical supervised learning task is classification. The spam filter is a good example of such a solution: it is trained with many example emails along with their class (spam/ham) and it must learn how to classify new unseen emails.

Another typical task is to predict a target numerical value, this is called regression. We have certain predictors such as the mileage, age, brand that are called features, and we must predict the target (price).

Classification is used to predict categorical (discrete) data and Regression is used to predict continuous data.

The data used to predict the value are called predictor/feature variable(s) and the value to be predicted is the target variable.

Some regression algorithms can be used for classification as well, for example Logistic Regression is used for classification as it gives us the probability for a data point belonging to a specific class.

Examples of supervised learning algorithms are as follows: -

1. k-Nearest Neighbors
2. Linear Regression
3. Logistic Regression
4. Support Vector Machine
5. Decision Tree and Random Forest
6. Neural Network

#### 4.1.2 Unsupervised Learning

In unsupervised learning, the major difference is that the training set we feed to the algorithm does not have labels i.e. it is unlabelled data. The system hence learns without supervision.

Some important unsupervised learning algorithms are as follows: -
1. Clustering (K-means, DBSCAN, Hierarchical Cluster Analysis - HCA)
2. Anomaly Detection and novelty detection (One-class SVM, Isolation Forest)
3. Visualisation and Dimensionality Reduction (Principal Component Analysis - PCA, Kernel PCA, Locally Linear Embedding - LLE, t-Distributed Stochastic Neighbour Embedding - t-SNE)
4. Association Rule Learning (Apriori, Eclat)

**Clustering**: Let's say you have a large amount of unlabelled data and that data is of various people who visit your blog site, you could use a clustering algorithm to separate those people into groups and at no point in the process would you have to intervene or decide how the people are seperated into groups, rather the model does that itself by finding patterns in the data. For example, it could find how some users like posts on comic books, movies, e.t.c. Hierarchical Clustering can be used to then further divide people in a group into smaller subgroups which you can then use to further tailor what the user sees.

**Visualisation**: These algorithms are also good examples of unsupervised learning: you feed them a lot of complex and unlabelled data, and they output 2D and 3D representation of your data that can be easily plotted. They try to preserve as much structure as possible (try to keep different clusters in the input space from overlapping in the visualisation) so that you can understand hpw the data is organised and perhaps identify unsuspected patterns.

**Dimensionality Reduction**: A related task is dimensionality reduction in which the goal is to simplify the data without losing too much information, for example if you have the data of a car, and you have attributes for the mileage and age, you could say that these are related and so the algorithm will reduce it to a single attribute indicating the cars wear and tear. It is good practice putting data through such an algorithm before feeding it to another algorithm (such as a classification algorithm), this reduces time required to train and could also improve performance.

**Anomaly Detection**: if we want to detect some anomalies, such as unusual credit card transactions to prevent fraud, or certain manufacturing defects, or remove outliers in data before feeding dataset to another algorithm we use this method. The system is shown **mostly normal** instances during training so that it can learn to recognise them, thus when it sees a new instance it can tell whether this instance is normal or an anomaly.

**Novelty Detection**: if we want to detect new instances that are different from the normal ones we use this method, the major difference is that the training dataset **must not** contain any of the new instances unlike in anomaly detection where some anomalies could be present in the dataset. For example if you want to detect pictures of a pug as a new instance your dataset should not contain any pictures of pugs rather it should have other dogs pictures otherwise it won't treat the pug as a new instance.

**Association Rule Learning**: in this task, the goal is to dig into large amounts of data and discover interesting relations between attributes. E.g. if you own a supermarket, running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also buy steak. These items can then be kept close to each other.

#### 4.1.3 Semi-supervised Learning

Labelling data can be a very time-consuming task, often datasets will have a large number of unlabelled instances and a few labelled instances. There are however some algorithms that can deal with data that is partially labelled. This is called semi supervised learning. For example the photos' app on your phone could recognise that person A appears in certain photos and person B appears in certain photos this is an example of dealing with unlabelled data i.e. clustering, all you need to do is give one label then to ech person.
Most semi-supervised learning algorithms are combinations of supervised and unsupervised algorithms e.g. deep belief networks (DBNs) are based on unsupervised components called restricted Boltzmann machines (RBMs) stacked on top of each other. RBMs are trained sequentially in an unsupervised manner and then the whole system is fine-tuned using supervised learning techniques.

#### 4.1.4 Reinforcement Learning

In reinforcement learning the learning system is called an **agent**, it observes the **environment** , selects and performs **actions**, gets **rewards** or **penalties** (negative rewards) based on the action it takes. It must then learn by itself what is the best strategy (based on rewards/penalties), this is called a **policy** to get the most reward over time. A policy defines what action an agent should choose when in a particular situation.
For example, you have AlphaZero that learned how to play chess by playing games against itself, another example of this would be DeepMind's AlphaGo program that learned how to play Go using reinforcement learning.

### 4.2 Batch and Online Learning

Another criteria used to classify machine learning is whether or not system can learn incrementally from a stream of incoming data.

#### 4.2.1 Batch Learning

In batch learning, the system is **incapable of learning incrementally**; it must be trained using all the available data. This generally takes a lot of time and computing resources and is typically done offline hence it is also called offline learning.
First the system is trained on the complete set of data after which it is launched into production and does not learn anymore, instead it just applies what it has already learned. If you want a batch learning system to learn new data then it needs to learn about all the data from scratch we cannot just update it with the new data. This approach is very time-consuming and during the training period the system will be offline. You could just to update the system to new data as and when needed by scheduling a time period when it will update.
If the amount of data is large and the data often changes i.e. the system needs to keep adapting to new data then a more reactive solution is needed in these cases it is better to use a system that can incrementally learn.

#### 4.2.2 Online Learning

In online learning the system can **incrementally learn**  by feeding it data batches sequentially, either individually or in small groups called **mini-batches**. Each learning step is fast and cheap, so the system can learn about new data on the fly as it arrives.
Online learning is great for systems that receive data as a continuous flow (e.g. stock prices) and need to adapt to change rapidly or autonomously. It is a good option if you have limited computing resources as the data is sent in mini-batches, and after it is learnt it is no longer needed and is discarded, this is unless you want to save it so that in a particular scenario you could roll back the system.
Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machines main memory (this is called out-of-core training). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all the data (out-of-core training is usually done offline however it is still incremental training).
One important feature of online learning algorithms is the **learning rate** i.e. how fast the algorithms adapts to new and changing data. If a high learning rate is set it adapts to new data quickly but also tend to forget old data (spam filter will then forget about older spam data which we don't want). A low learning rate system will have more inertia that is it will learn slower, but it will be less sensitive to noise in the new data or to outliers.
The big challenge with online learning is that if bad data is fed to the system the performance will gradually decline, if it's a live system the clients will notice, to reduce the risk you need to monitor the system closely and switch off learning or revert ot a previously working state if you detect a drop in performance. You may also want to monitor input data and react to abnormal data (using anomaly detection algorithm).

### 4.3 Instance Based vs Model Based

Another criteria to categorize Machine Learning algorithms is by how they generalise. Most ML tasks are about making prediction. This means that they are given a number of training examples, the system needs to be able tp make good predictions for (generalise to) examples it has never seen before. Having a good performance measure on the training data is good but insufficient; the true goal is to perform well on new instances. 
There are two main approaches to generalization: instance based and model based.

#### 4.3.1 Instance Based Learning

The concept of instance based learning is very simple, it is simply a method where the algorithm learns the example by heart. If you were to create a spam filter in this manner it would flag all the spam emails that are identical to the spam one's present in the dataset. This is not a very good solution as it only flags identical mails.
Another method would be to use a measure of similarity using which we could find emails that are similar to the spam emails and hence flag them. This could be done by comparing words and the frequency of words and other patterns. THis method is better at generalising.
Hence, instance based learning is learning by heart and then generalising to new cases by using a similarity measure to compare them to learned examples or a subset of them.

#### 4.3.2 Model Based Learning

Another way to generalise from a set of examples is to build a model of these examples and then use the model to make predictions. This is called model based learning. If we notice a linear relationship between attribute and target we decide to use a linear function this is called model selection. The model will have certain parameters. The value of the parameters are decided using a specific performance measure which we need to specify.
You can either define a utility function(fitness function) that measures how good a model is, or you can define a cost function that measures how bad a model is. For Linear Regression we use a cost function that measures distance between models predictions and training examples, the goal is to minimise the distance(i.e. minimise the cost function).
This is where the algorithm does its task, you feed it with the data, and it finds the parameters that ,ake tje model fit best to your data, After this you can use the model to make predictions. In summary, you must study the data, select a model, train it on training data, and apply the model to mae predictions on new cases. This is the typical workflow of an ML project. 

## 5. Main Challenges of Machine Learning

The two main things that can go wrong in machine learning is selecting a **"bad algorithm"** and using **"bad data"**.  Examples of these are as follows: -

### 5.1 Examples of Bad Data

#### 5.1.1 Insufficient Quantity of Training Data

One of the major drawbacks on machine learning is that it requires a very large amount of data to work effectively. For example if human can learn what a car is by looking at a few pictures but a machine learning algorithm will require thousands of examples. Sometimes it is hard to find such large amounts of data and not having sufficient quantities of data is one of the major reasons as to why ML may not give adequate results. 
Some studies have even showed that the amount of data is more important than the algorithm that is chosen. However, finding cheap extra training data is hard so this we must try to optimize everything including finding a good algorithm.

#### 5.1.2 Nonrepresentative Training Data

It is important to use training data that is representative of the new cases you want to generalise to, if the relationship of the features in training to the target is vastly different in the new cases then the model will not perform properly. 
Solving this is harder than it sounds: if the sample is too small you will have sampling noise (i.e. nonrepresentative data as a result of chance) but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

#### 5.1.3 Poor Quality Data

If the training data is full of errors, outliers and noise (e.g. due to poor quality measurements), it will make it harder for the system to detect underlying patterns and performance of the system will suffer. It is often well worth the time to spend cleaning up the data.
A large amount of data scientist time goes in performing this task. The following are examples of when you would want to clean up training data: -
1. If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually.
2. If some instances are missing a few features (e.g. 5% of customers didn't give their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g. with the median age), or train one model with the feature and one model without it.

#### 5.1.4 Irrelevant Features

An important thing to remember is "garbage in, garbage out". Your system will only be capable of learning if the training data contains enough relevant features and not too mny irrelevant ones, A critical part of ML is coming up with a good set of features to train on. This process, called feature engineering, involves the following steps:
1. **Feature selection**: selecting the most useful features to train on among existing features
2. **Feature extraction:** combining existing features to produce a more useful one, as seen earlier dimensionality reduction algorithms can help with this.
3. **Creating new features** by gathering new data

### 5.2 Examples of Bad Algorithm

#### 5.2.1 Overfitting the Training Data

Overfitting is a scenario where your algorithm performs very well on training data but not on testing data. This means that the algorithm cannot generalise well to unseen cases. This usually happens when the algorithm is too complex, and it finds patters in the noise rather than in the actual data itself. To prevent this from happening we can perform the following steps:
1. Simplify the model to one with fewer parameters
2. Gather more training data
3. Reduce the noise in the training data (e.g. fix data errors and remove outliers)

Constraining a model to make it simpler and reduce the risk of overfitting is called regularisation. We need to find the right balance between fitting the training data well and keeping the model simple enough to ensure it will generalise well. The amount of regularisation to apply during learning can be controlled using a hyperparameter. 
If set to a large value you will get almost a flat model (a slope close to zero); the model will almost certainly not overfit the data but, it will be less likely to find a good solution.

#### 5.2.2 Underfitting the Training Data

Underfitting is the opposite of overfitting. It occurs when the model is too simple to learn the underlying structure of the data, For example a linear model of life satisfaction is prone to underfit; as reality is more complex than the model so its predictions are inaccurate even on training data. Here are some examples: -
1. Select a more powerful model, with more parameters
2. Feed better features to the learning algorithm (feature engineering)
3. Reduce the constraints on the model (e.g. reduce the regularisation hyperparameter).

## 6. Testing and Validating

The only way to know how well a model will generalise to new cases is to actually try it out on new cases. One way to do this is to put the model into production and see how it performs. This works well but if your model is bad the users will notice and complain.
A better option is to split the dataset into two sets, the training and test set. As implied you train the model using the training set and test it using the test set. The error rate on the new cases is called the generalisation error (or out-of-sample error) and by evaluating the model on test set you get an estimate of this error.This value tells you how well your model will perform on instances it has never seen before.
If training error is low and generalisation error is high it means your model is overfitting the training data. it is common to use 80% of the data for training and 20% for testing. This however depends on size of the dataset. If the dataset has millions of samples even 1% for testing would be enough.

### 6.1 Hyperparameter Tuning and Model Selection

In a scenario where you have multiple models, and you are not sure which model to choose for further development, it is best to evaluate those two models and compare their performances to each other. For example, if you have a linear model and a polynomial model we should train both of them and then evaluate them on the test set the model that does better will then be chosen for further development.
After choosing this model we now need to perform regularisation by setting the hyperparameters of this model in such a way that the performance of the model increases, for example if there are 10 possible values for a hyperparameter then we must use all 10 as 10 separate models and find which one performs the best, we will then choose to use that one. 
Once we launch this model into production we might get a high rate of error even though it was performing well on the test data, the reason for this is we have ended up choosing a model that is too fine-tuned to the test data this is called holdout validation as we have fine-tuned parameters to give the best result for that particular set.
We can solve this by using another set called the validation set that is separate from the training set, we train our data on the train set, then we regularise it on the validation set and fine tune it according to this set, you then train the best model on the full train set and validation set, and then we evaluate it on the test set to get the final generalisation error, as a result test set yet acts as unseen data.
If the validation set is too small, then the model evaluations will be imprecise: you may end up selecting a suboptimal model. If too large then the remaining training set is much smaller than the full training set, it is not ideal to compare candidate models trained ona  much smaller training set. One way to solve this problem is to use cross-validation using many small validation sets.
Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of the model, you get a much more accurate measure of its performance. There is a drawback however, the training time is multiplied by the number of validation sets.

### 6.2 Data Mismatch

In some cases the data you have in your dataset may be nonrepresentative of the actual data you will get. For example, if you use pictures of flowers downloaded from the internet and the user takes a photo using the apps camera and wants to make a prediction using it is possible that the photo may not be of the same type as the data you have. In the scenario it is a data mismatch.
As a result of this you won't know whether your model is performing poorly because the model is bad or because there is a mismatch of data. A solution is to create a hold out set called the train-dev set. After the model is trained on the training set you can evaluate it on the train dev set. If it performs well, that meas it is not overfitting, if it performs poorly on the validation set then the problem must be coming from data mismatch. 
You can try to tackle this by preprocessing the web images to make them look more like images taken by the mobile app, and then retrain the model. Conversely, if the model performs poorly on the train-dev set then it has overfitted the training set, so you should try to simplify or regularize the model, get more data, and clean up the training data (this method was suggested by Andrew Ng).