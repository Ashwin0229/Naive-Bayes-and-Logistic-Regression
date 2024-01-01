Option 1:

To execute,
Run the algorithm.py file with arguments as below for each model and representation

Multinomial Naive Bayes on Bag of words:
	python algorithm.py MULTINOMIALNB 1
	python algorithm.py MULTINOMIALNB 2
	python algorithm.py MULTINOMIALNB 3

Discrete Naive Bayes on Bernoulli's model:
	python algorithm.py DISCRETENB 1
	python algorithm.py DISCRETENB 2
	python algorithm.py DISCRETENB 3


Logistic Regression (Bag of words & Bernoulli):
	python algorithm.py LR 1
	python algorithm.py LR 2
	python algorithm.py LR 3

SGD Classifier (Bag of words & Bernoulli):
	python algorithm.py SGDClassifier 1
	python algorithm.py SGDClassifier 2
	python algorithm.py SGDClassifier 3



Option 2:

To run in google colab,

extract the colab_files.zip

in colab
	1. open notebook (algorithms.ipynb)
	2. Load the Datasets.zip in the Files section of the colab
	3. Run the notebook from starting

Depending on which algorithm to use, comment and uncomment the strings in main function