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

A computer program is said to learn from experience E with respect to some task T and some performance measure P, its performance on T, as measured bu P, improves with experience E.

A spam filter is a Machine Learning program that given examples of spam emails and regular emails can learn to flag spam, The examples that the system uses to learn are called the training set.
Each training example is called a training instance (or sample). In this case task T is to flag spam for new emails, experience E is training data and performace measurr P needs to be defined. 
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
2. You would write a detection algorithm for each of the patterns you noticed and if a certain amount of ptterns are noticed it is flagged as spam.
3. You would test the program and repeat steps 1 and 2 till the algorithm is good enough to be used.

Since the problem is difficult it is likely that the program will become a long list of complex rules which will be hard to maintain, as opposed to a ML solution where the algorithm will learn automatically which words and phrases are common to be found in spam and is hence a good spam detector.
Also, if attackers notice some words like "4U" are getting blocked they might start using "For u" instead and hence we would have to update our algorithm, instead in an ML solution it would automatically learn that "For u" is also being found in spam.

E.g. 2) In a voice recognition program if we are trying to classify words "apple" and "orange", the steps to do this in a traditional approach will be highly complex as we will have a large number of individual rules for each word such as the pitch and there will be a lot of other variables involved. To avoid this complexity a ML solution is much better at learning patters from the data and is not as complex.

We can also use the patterns found machine learing algorithms to learn more about the data, e.g. in the spam filter we could find what words are often used in spam mails.


## 3. Applications of Machine Learning

Some concrete examples of machine learning and techniques that can be used to tackle them are as follows: -

1. Analyzing images of products on a production line ot automatically classify them: this is image classification typically performed using convolutional neural networks.
2. Detecting tumours in brain scans: this is semantic segmentation, where ech pixel in the image is classified (as we want to determine the exact location and shape of the tumor), typically using CNN's.
3. Automatically classifying news articles: this is natural language processing (NLP) and more specifically text classification which can be tackled using reccurent neural networks (RNN) and convolutional neural network (CNN)
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

Another typical task is to predict a target numerical value, this is called regression. We have certain predictors such as the mileage, age, brand that are called features and we must predict the target (price).

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

**Clustering**: Let's say you have a large amount of unlabelled data and that data is of various people who visit your blog site, you could use a clustering algorithm to separate those people into groups and at no point in the process would you have to intervene or decide how the people are seperated into groups, rather the model does that itself by finding patterns in the data. For example it could find how some users like posts on comic books, movies, e.t.c. Hierarchical Clustering can be used to then further divide people in a group into smaller subgroups which you can then use to further tailor what the user sees.

**Visualisation**: These algorithms are also good examples of unsupervised learning: you feed them a lot of complex and unlabelled data and they output 2D and 3D representation of your data that can be easily plotted. They try to preserve as much structure as possible (try to keep different clusters in the input space from overlapping in the visualisation) so that you can understand hpw the data is organised and perhaps identify unsuspected patterns.

**Dimensionality Reduction**: A related task is dimensionality reduction in which the goal is to siplify the data without losing too much information, for example if you have the data of a car, and you have attributes for the mileage and age, you could say that these are related and so the algorithm will reduce it to a single attribute indicating the cars wear and tear. It is good oractice to put data through such an algorithm before feeding it to another algorithm (such as a classification algorithm), this reduces time required to train and could also improve performance.

**Anomaly Detection**: if we want to detect some anomalies, such as unusual credit card transactions to prevent fraud, or certain manufacturing defects, or remove outliers in data before feeding dataset to another algorithm we use this method. The system is shown **mostly normal** instances during training so that it can learn to recognise them, thus when it sees a new instance it ca tell whether this instance is normal or an anomaly.

**Novelty Detection**: if we want to detect new instances that are different from the normal ones we use this method, the major difference is that the training dataset **must not** contain any of the new instances unlike in anomaly detection where some anomalies could be present in the dataset. For example if you want to detect pictures of a pug as a new instance your dataset should not contain any pictures of pugs rather it should have other dogs pictures otherwise it won't treat the pug as a new instance.

**Association Rule Learning**: in this task, the goal is to dig into large amounts of data and discover interesting relations between attriibutes. E.g. if you own a supermarket, running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also buy steak. These items can then be kept close to each other.

#### 4.1.3 Semi-supervised Learning



#### 4.1.4 Reinforcement Learning


### 4.2 Batch and Online Learning


#### 4.2.1 Batch Learning


#### 4.2.2 Online Learning


### 4.3 Instance Based vs Model Based


#### 4.3.1 Instance Based Learning



#### 4.3.2 Model Based Learning


## 5. Main Challenges of Machine Learning


## 6. Testing and Validating


