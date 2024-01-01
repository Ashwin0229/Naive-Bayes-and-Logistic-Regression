'''
Name   : Ashwin Sai C
Course : ML - CS6375-003
Title  : Mini Project 1
Term   : Fall 2023

'''

import nltk
import os
import sys
import itertools
import re
from collections import Counter
import numpy as np
import math
import pandas as pd
from scipy.special import expit
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# warnings.filterwarnings('ignore')

#Dataset 1
train1_path         = "Dataset1\\enron1_train\\enron1\\train"
test1_path          = "Dataset1\\enron1_test\\enron1\\test"
train1_path_ham     = "Dataset1\\enron1_train\\enron1\\train\\ham"
train1_path_spam    = "Dataset1\\enron1_train\\enron1\\train\\spam"
test1_path_ham      = "Dataset1\\enron1_test\\enron1\\test\\ham"
test1_path_spam     = "Dataset1\\enron1_test\\enron1\\test\\spam"

#Dataset 2
train2_path         = "Dataset2\\enron2_train\\train"
test2_path          = "Dataset2\\enron2_test\\test"
train2_path_ham     = "Dataset2\\enron2_train\\train\\ham"
train2_path_spam    = "Dataset2\\enron2_train\\train\\spam"
test2_path_ham      = "Dataset2\\enron2_test\\test\\ham"
test2_path_spam     = "Dataset2\\enron2_test\\test\\spam"

#Dataset 3
train3_path         = "Dataset3\\enron4_train\\enron4\\train"
test3_path          = "Dataset3\\enron4_test\\enron4\\test"
train3_path_ham     = "Dataset3\\enron4_train\\enron4\\train\\ham"
train3_path_spam    = "Dataset3\\enron4_train\\enron4\\train\\spam"
test3_path_ham      = "Dataset3\\enron4_test\\enron4\\test\\ham"
test3_path_spam     = "Dataset3\\enron4_test\\enron4\\test\\spam"


def process_dataset(dataset_path):  
    
    file_count         = 0
    complete_tokens    = []
    complete_word_freq = {}
    filename_list      = []
    y                  = []

    for path, subdirs, files in os.walk(dataset_path):
        for name in files:
            file_count += 1                 
            [merged_list, word_freq] = tokenize_email(path+"\\"+name)
            
            complete_tokens.append(merged_list) 
            complete_word_freq[name] = word_freq
            filename_list.append(name)              

    complete_merged_list  = list(itertools.chain.from_iterable(complete_tokens))
    nonunique_merged_list = complete_merged_list 
    complete_merged_list  = set(complete_merged_list)   

    [bag_of_words_vector, bernoulli_vector]  = feature_matrix_generation(filename_list,complete_merged_list,complete_word_freq)

    for file_name in filename_list:
        if "ham.txt" in file_name:
            y.append(0)
        elif "spam.txt" in file_name:
            y.append(1)
        else:
            print(file_name)

    Y_count = Counter(y)

    print("DataSet Path        : ",dataset_path)
    print("Ham -0, Spam -1,    : ",Y_count) 
    print("Bag of words length : ",len(bag_of_words_vector))
    print("Bernoulli's length  : ",len(bernoulli_vector))       
    
    return [bag_of_words_vector, bernoulli_vector, nonunique_merged_list, y]

def feature_matrix_generation(filename_list,complete_merged_list,complete_word_freq):
    
    bag_of_words       = []
    bernoulli_words    = []

    for name in filename_list:  
        temp_list   = []
        temp_list_2 = []        

        for word in complete_merged_list:
            try:
                if word in complete_word_freq[name].keys():
                    temp_list.append(complete_word_freq[name][word])
                    temp_list_2.append(1)
                else:
                    temp_list.append(0)
                    temp_list_2.append(0)
            except Exception as e:
                print(e)                
        
        bag_of_words.append(temp_list)
        bernoulli_words.append(temp_list_2)

    return [bag_of_words, bernoulli_words]

def tokenize_email(filename):
    f=open(filename,'r',encoding='cp437')
    raw=f.readlines()   
    token_set = []
    for line in raw:
        tokens = nltk.word_tokenize(line)               
        token_set.append(tokens)    
    merged_list = list(itertools.chain.from_iterable(token_set))        
    counts = Counter(merged_list)   

    return [merged_list, counts]    

def feature_extraction(algorithm_type, dataset_used):   
    
    print("---Extracting all Datasets---")
    [train_bag_of_words_vector_1, train_bernoulli_vector_1, wordlist_train_1, y_train1]                      = process_dataset(train1_path) 
    [test_bag_of_words_vector_1, test_bernoulli_vector_1, wordlist_test_1, y_test1]                          = process_dataset(test1_path) 

    [train_bag_of_words_vector_2, train_bernoulli_vector_2, wordlist_train_2, y_train2]                      = process_dataset(train2_path) 
    [test_bag_of_words_vector_2, test_bernoulli_vector_2, wordlist_test_2, y_test2]                          = process_dataset(test2_path) 

    [train_bag_of_words_vector_3, train_bernoulli_vector_3, wordlist_train_3, y_train3]                      = process_dataset(train3_path) 
    [test_bag_of_words_vector_3, test_bernoulli_vector_3, wordlist_test_3, y_test3]                          = process_dataset(test3_path)

    [train_bag_of_words_vector_1_ham, train_bernoulli_vector_1_ham, wordlist_train_1_ham, y_train1_ham]      = process_dataset(train1_path_ham) 
    [train_bag_of_words_vector_1_spam, train_bernoulli_vector_1_spam, wordlist_train_1_spam, y_train1_spam]  = process_dataset(train1_path_spam) 

    [train_bag_of_words_vector_2_ham, train_bernoulli_vector_2_ham, wordlist_train_2_ham, y_train2_ham]      = process_dataset(train2_path_ham) 
    [train_bag_of_words_vector_2_spam, train_bernoulli_vector_2_spam, wordlist_train_2_spam, y_train2_spam]  = process_dataset(train2_path_spam) 

    [train_bag_of_words_vector_3_ham, train_bernoulli_vector_3_ham, wordlist_train_3_ham, y_train3_ham]      = process_dataset(train3_path_ham) 
    [train_bag_of_words_vector_3_spam, train_bernoulli_vector_3_spam, wordlist_train_3_spam, y_train3_spam]  = process_dataset(train3_path_spam) 

    print("\n---Extraction done---\n\n")

    datatset_wordCount_1     = {"ham": train_bag_of_words_vector_1_ham, "spam": train_bag_of_words_vector_1_spam}
    dataset_words_1          = {"ham": wordlist_train_1_ham, "spam": wordlist_train_1_spam, "both": wordlist_train_1}

    datatset_wordCount_2     = {"ham": train_bag_of_words_vector_2_ham, "spam": train_bag_of_words_vector_2_spam}
    dataset_words_2          = {"ham": wordlist_train_2_ham, "spam": wordlist_train_2_spam, "both": wordlist_train_2}

    datatset_wordCount_3     = {"ham": train_bag_of_words_vector_3_ham, "spam": train_bag_of_words_vector_3_spam}
    dataset_words_3          = {"ham": wordlist_train_3_ham, "spam": wordlist_train_3_spam, "both": wordlist_train_3}

    datatset_wordCount_DNB_1 = {"ham": train_bernoulli_vector_1_ham, "spam": train_bernoulli_vector_1_spam, "both": train_bernoulli_vector_1}
    dataset_words_DNB_1      = {"ham": wordlist_train_1_ham, "spam": wordlist_train_1_spam, "both": wordlist_train_1}

    datatset_wordCount_DNB_2 = {"ham": train_bernoulli_vector_2_ham, "spam": train_bernoulli_vector_2_spam, "both": train_bernoulli_vector_2}
    dataset_words_DNB_2      = {"ham": wordlist_train_2_ham, "spam": wordlist_train_2_spam, "both": wordlist_train_2}

    datatset_wordCount_DNB_3 = {"ham": train_bernoulli_vector_3_ham, "spam": train_bernoulli_vector_3_spam, "both": train_bernoulli_vector_3}
    dataset_words_DNB_3      = {"ham": wordlist_train_3_ham, "spam": wordlist_train_3_spam, "both": wordlist_train_3}

    Dataset_Path             = {

                                    "Dataset1":
                                                {
                                                    "train_path"      : train1_path,
                                                    "test_path"       : test1_path,
                                                    "train_path_ham"  : train1_path_ham,
                                                    "train_path_spam" : train1_path_spam,
                                                    "test_path_ham"   : test1_path_ham,
                                                    "test_path_spam"  : test1_path_spam
                                                },
                                    "Dataset2":
                                                {
                                                    "train_path"      : train2_path,
                                                    "test_path"       : test2_path,
                                                    "train_path_ham"  : train2_path_ham,
                                                    "train_path_spam" : train2_path_spam,
                                                    "test_path_ham"   : test2_path_ham,
                                                    "test_path_spam"  : test2_path_spam
                                                },
                                    "Dataset3":
                                                {
                                                    "train_path"      : train3_path,
                                                    "test_path"       : test3_path,
                                                    "train_path_ham"  : train3_path_ham,
                                                    "train_path_spam" : train3_path_spam,
                                                    "test_path_ham"   : test3_path_ham,
                                                    "test_path_spam"  : test3_path_spam
                                                }
                               }

    if algorithm_type == "MULTINOMIALNB":       
        if dataset_used == "1":
            ExecuteMULTINOMIALNB(train_bag_of_words_vector_1, datatset_wordCount_1, dataset_words_1, Dataset_Path["Dataset1"], y_test1)    #Dataset1
        elif dataset_used  == "2":
            ExecuteMULTINOMIALNB(train_bag_of_words_vector_2, datatset_wordCount_2, dataset_words_2, Dataset_Path["Dataset2"], y_test2)    #Dataset2
        elif dataset_used == "3":
            ExecuteMULTINOMIALNB(train_bag_of_words_vector_3, datatset_wordCount_3, dataset_words_3, Dataset_Path["Dataset3"], y_test3)    #Dataset3
        else:
            print("Select proper Dataset (1,2,3)")

    elif algorithm_type == "DISCRETENB":
        if dataset_used == "1":
            ExecuteDISCRETENB(train_bernoulli_vector_1, datatset_wordCount_DNB_1, dataset_words_DNB_1, Dataset_Path["Dataset1"], y_test1)  #Dataset1
        elif dataset_used == "2":
            ExecuteDISCRETENB(train_bernoulli_vector_2, datatset_wordCount_DNB_2, dataset_words_DNB_2, Dataset_Path["Dataset2"], y_test2)  #Dataset2
        elif dataset_used == "3":
            ExecuteDISCRETENB(train_bernoulli_vector_3, datatset_wordCount_DNB_3, dataset_words_DNB_3, Dataset_Path["Dataset3"], y_test3)  #Dataset3
        else:
            print("Select proper Dataset (1,2,3)")

    elif algorithm_type == "LR":
        if dataset_used == "1":
            print("..")
            print("Set 1")           
            print("Bag of Words:")
            APPLYLR(train_bag_of_words_vector_1, y_train1, wordlist_train_1, test_bag_of_words_vector_1, y_test1, wordlist_test_1, 0.0001, 1.2, 1000, 1)
            print("\nBernoulli's:")
            APPLYLR(train_bernoulli_vector_1, y_train1, wordlist_train_1, test_bernoulli_vector_1, y_test1, wordlist_test_1, 0.0001, 1.2, 1000, 1)
        elif dataset_used == "2":
            print("..")
            print("Set 2")
            print("Bag of Words:")
            APPLYLR(train_bag_of_words_vector_2, y_train2, wordlist_train_2, test_bag_of_words_vector_2, y_test2, wordlist_test_2, 0.0001, 1.2, 1000, 1)
            print("\nBernoulli's:")
            APPLYLR(train_bernoulli_vector_2, y_train2, wordlist_train_2, test_bernoulli_vector_2, y_test2, wordlist_test_2, 0.0001, 1.2, 1000, 1)
        elif dataset_used == "3":
            print("..")
            print("Set 3")
            print("Bag of Words:")
            APPLYLR(train_bag_of_words_vector_3, y_train3, wordlist_train_3, test_bag_of_words_vector_3, y_test3, wordlist_test_3, 0.000001, 0.8, 1000, 1)
            print("\nBernoulli's:")
            APPLYLR(train_bernoulli_vector_3, y_train3, wordlist_train_3, test_bernoulli_vector_3, y_test3, wordlist_test_3, 0.000001, 0.8, 1000, 1)

    elif algorithm_type == "SGDClassifier":
        if dataset_used == "1":
            print("..")
            print("Bag of Words:")
            TRAINSGDCLASSIFIER(train_bag_of_words_vector_1, y_train1, wordlist_train_1, test_bag_of_words_vector_1, y_test1, wordlist_test_1)
            print("\nBernoulli's:")
            TRAINSGDCLASSIFIER(train_bernoulli_vector_1, y_train1, wordlist_train_1, test_bernoulli_vector_1, y_test1, wordlist_test_1)
        elif dataset_used == "2":
            print("..")
            print("Bag of Words:")
            TRAINSGDCLASSIFIER(train_bag_of_words_vector_2, y_train2, wordlist_train_2, test_bag_of_words_vector_2, y_test2, wordlist_test_2)
            print("\nBernoulli's:")
            TRAINSGDCLASSIFIER(train_bernoulli_vector_2, y_train2, wordlist_train_2, test_bernoulli_vector_2, y_test2, wordlist_test_2)
        elif dataset_used == "3":
            print("..")
            print("Bag of Words:")
            TRAINSGDCLASSIFIER(train_bag_of_words_vector_3, y_train3, wordlist_train_3, test_bag_of_words_vector_3, y_test3, wordlist_test_3)
            print("\nBernoulli's:")
            TRAINSGDCLASSIFIER(train_bernoulli_vector_3, y_train3, wordlist_train_3, test_bernoulli_vector_3, y_test3, wordlist_test_3)

def ExecuteMULTINOMIALNB(words_vector, datatset_wordCount, dataset_words, dataset_path, Y):

    print("Executing Mutlinomial NB..")
    print(dataset_path)

    [C, V, prior, conditional_prob] = TRAINMULTINOMIALNB(len(words_vector), datatset_wordCount, dataset_words)

    print("Training done")

    print("Testing...")

    count = 0
    total_count = 0
    Y_Pred = []
    for path, subdirs, files in os.walk(dataset_path["test_path"]):
        for name in files:          
            score = APPLYMULTINOMMIALNB(C, V, prior, conditional_prob, path+"\\"+name)
            # print(name, ": ", score)  
            if score == "ham":
                count += 1
                Y_Pred.append(0)
            else:
                Y_Pred.append(1)
    
    print('\nPrecision: %.3f' % precision_score(Y, Y_Pred))
    print('Recall: %.3f' % recall_score(Y, Y_Pred))
    print('F1-Score: %.3f' % f1_score(Y, Y_Pred))
    print('Accuracy: %.3f' % accuracy_score(Y, Y_Pred))
       
def ExecuteDISCRETENB(words_vector, datatset_wordCount, dataset_words, dataset_path, Y):   

    print("Executing Discrete NB..")
    print(dataset_path)

    [C, V, prior, conditional_prob] = TRAINDISCRETENB(len(words_vector), datatset_wordCount, dataset_words)

    print("Training done")

    print("Testing...")

    count       = 0
    total_count = 0
    Y_Pred      = []

    for path, subdirs, files in os.walk(dataset_path["test_path"]):
        for name in files:          
            score = APPLYDISCRETENB(C, V, prior, conditional_prob, path+"\\"+name)
            # print(name, ": ", score)  
            total_count += 1
            if score == "ham":
                count += 1
                Y_Pred.append(0)
            else:
                Y_Pred.append(1)
    

    print('\nPrecision: %.3f' % precision_score(Y, Y_Pred))
    print('Recall: %.3f' % recall_score(Y, Y_Pred))
    print('F1-Score: %.3f' % f1_score(Y, Y_Pred))
    print('Accuracy: %.3f' % accuracy_score(Y, Y_Pred))       

def TRAINMULTINOMIALNB(NumofDocuments, datatset_wordCount, dataset_words):
    V = set(dataset_words["both"])
    N = NumofDocuments
    C = ["ham","spam"]
    prior = {}
    conditional_prob = {}

    for c in C:
        print("Training Class : ",c)
        Nc = len(dataset_words[c])
        N  = len(dataset_words["both"])
        prior[c] = Nc/N
            
        conditional_prob[c] = {}
                
        for word_index,t in enumerate(set(dataset_words["both"])):
            if t in set(dataset_words[c]):
                # print(set(dataset_words[c]))              
                index = list(set(dataset_words[c])).index(t)
                word_sum_class = np.sum(datatset_wordCount[c],axis=0)
                Tct = word_sum_class[index]
                # print(Tct)
            else:
                Tct = 0


            conditional_prob[c][t] = (Tct + 1) / (sum(np.sum(datatset_wordCount[c],axis=0)) + len(V))

    # print(prior)
    # print(conditional_prob)

    return C, V, prior, conditional_prob

def APPLYMULTINOMMIALNB(C, V, prior, conditional_prob, test_doc):
    W,counts = tokenize_email(test_doc)
    score = {}

    for c in C:
        score[c] = math.log(prior[c])
        for t in set(W):
            try:
                score[c] = score[c] + math.log(conditional_prob[c][t])
            except Exception as e:
                # print("word not found : ",t)
                # print(e)
                pass

    return max(score, key=score.get)

def APPLYDISCRETENB(C, V, prior, conditional_prob, test_doc):
    Vd,counts_dummy = tokenize_email(test_doc)
    score = {}

    for c in C:
        score[c] = math.log(prior[c])
        for t in set(V):
            if t in Vd:
                score[c] = score[c] + math.log(conditional_prob[c][t])
            else:
                try:
                    score[c] = score[c] + math.log(1 - conditional_prob[c][t])
                except Exception as e:
                    print(e)
                    pass

    return max(score, key=score.get)

def TRAINDISCRETENB(NumofDocuments, datatset_wordCount, dataset_words):
    V = set(dataset_words["both"])
    N = NumofDocuments
    C = {"ham","spam"}
    prior = {}
    conditional_prob = {}
    print(C)
    for c in C:
        print("Training Class : ",c)
        Nc = len(dataset_words[c])
        N  = len(dataset_words["both"])
        prior[c] = Nc/N

        conditional_prob[c] = {}

        for word_index, t in enumerate(set(dataset_words["both"])):         
            if t in set(dataset_words[c]):
                index = list(set(dataset_words[c])).index(t)                            
                word_count_class = np.sum(datatset_wordCount[c], axis=0)
                Nct = word_count_class[index]
                # denominator = (np.sum(datatset_wordCount["both"],axis=0)[index] + len(datatset_wordCount["both"]))
            else:
                Nct = 0             

            conditional_prob[c][t] = (Nct + 1) / (Nc + 2)

    # print(conditional_prob)

    return C, V, prior, conditional_prob

def LR_Preprocessing2(x, y, wordlist_train, x_test, y_test, wordlist_test):
    
    common_columns    = list(set(wordlist_train) & set(wordlist_test))
    common_col_df     = pd.DataFrame(common_columns)

    x_df              = pd.DataFrame.from_records(x)
    x_test_df         = pd.DataFrame.from_records(x_test)
    x_df.columns      = list(set(wordlist_train))
    x_test_df.columns = list(set(wordlist_test))

    # print(x_df)
    # print(x_test_df)

    x_filtered        = x_df[common_columns]
    x_test_filtered   = x_test_df[common_columns]

    # print(x_filtered)
    # print(x_test_filtered)    

    X_Train = x_filtered.values.tolist()
    X_Test  = x_test_filtered.values.tolist()

    # print(np.array(X_Train).shape)
    # print(len(y))
    # print(np.array(X_Test).shape)
    # print(len(y_test))

    return X_Train, y, X_Test, y_test
    
def LOGISTICREGRESSION(weights, X, Y, lambda_value, iteration_count, learning_rate): 

    # row, col = X.shape
    # weights  = np.zeros(col)

    for count in range(0,iteration_count): 
        # print(count)
        z    = np.dot(X, weights)        
        pred = expit(z)        

        gradient_value = (np.dot(X.T, (Y - pred))) - ((lambda_value) * weights)
        weights = weights + (learning_rate * gradient_value)

    return weights

def APPLYLR(x, y, wordlist_train, x_test, y_test, wordlist_test, lambda_value, learning_rate, iteration_count, epochs):

    X_Train, Y_Train, X_Test, Y_Test = LR_Preprocessing2(x, y, wordlist_train, x_test, y_test, wordlist_test)   

    X        = np.array(X_Train)
    Y        = np.array(Y_Train)
    X_test   = np.array(X_Test)
    Y_test   = np.array(Y_Test)

    split_value = 0.7

    X_Train_70 = X[:round(len(X)*split_value)]
    Y_Train_70 = Y[:round(len(Y)*split_value)]

    X_Test_30  = X[round(len(X)*split_value):]
    Y_Test_30  = Y[round(len(Y)*split_value):]

    # ## model 3 old   
    # lambda_value    = 0.1   
    # learning_rate   = 0.41
    # iteration_count = 1000
    # epochs          = 1 
    ## model 3    
    # lambda_value    = 0.000001   
    # learning_rate   = 0.8
    # iteration_count = 1000
    # epochs          = 1  

    # model 1 & 2    
    # lambda_value    = 0.0001   
    # learning_rate   = 1.2
    # iteration_count = 1000
    # epochs          = 1    
    

    row, col = X_Train_70.shape
    weights  = np.zeros(col)
    # for epoch_count in range(epochs):   
    #     print("Epoch count : ",epoch_count)     
    #     updated_weights = LOGISTICREGRESSION(weights, X_Train_70, Y_Train_70, lambda_value, iteration_count, learning_rate)
    #     weights = updated_weights

    # print(weights)
    # accuracy(X_Test_30, Y_Test_30, weights)

    weights = LOGISTICREGRESSION(weights, X, Y, lambda_value, iteration_count, learning_rate)
    # print(weights)
    accuracy(X_Test, Y_Test, weights)

def accuracy(X, Y, weights):

    z     = np.dot(X, weights)  
    # pred  = 1 / (1 + np.exp(-cc))
    pred = expit(z)
    
    if len(pred) != len(Y):
        print("Length mismatch!")
        exit(0)
    
    # count = 0
    # for index, pred_value in enumerate(pred):
    #     if pred_value == Y[index]:
    #         count += 1
    # accuracy_value = (count / len(pred)) * 100

    print('\nPrecision: %.3f' % precision_score(Y, np.round(abs(pred))))    
    print('Recall: %.3f' % recall_score(Y, np.round(abs(pred))))
    print('F1-Score: %.3f' % f1_score(Y, np.round(abs(pred))))
    print('Accuracy: %.3f' % accuracy_score(Y, np.round(abs(pred))))

    # print("Accuracy : ", accuracy_value)

def TRAINSGDCLASSIFIER(x, y, wordlist_train, x_test, y_test, wordlist_test):

    X_Train, Y_Train, X_Test, Y_Test = LR_Preprocessing2(x, y, wordlist_train, x_test, y_test, wordlist_test)   

    X = np.array(X_Train)
    Y = np.array(Y_Train)
    param_grid = {
                    'loss'     : ['log_loss'], # Different loss functions
                    'penalty'  : ['l2'],       # Regularization type
                    # 'alpha'    : [0.000001],      # Regularization strength Dataset 3
                    'alpha'    : [0.0001],      # Regularization strength Dataset 1,2
                    'max_iter' : [1000],     # Maximum number of iterations
                 }    

    clf = SGDClassifier()
    grid_search = GridSearchCV(clf, scoring='accuracy', param_grid=param_grid)

    grid_search.fit(X, Y)
    Y_Pred = grid_search.predict(X_Test)
        
    best_params = grid_search.best_params_
    best_sgd_clf = grid_search.best_estimator_

    sgd_pred = best_sgd_clf.predict(X_Test)

    count = 0
    for i,pred in enumerate(sgd_pred):
        if pred == Y_Test[i]:
            count += 1            

    # print("Accuracy : ",round((count/len(Y_Test))*100,2), "%")

    print('\nPrecision: %.3f' % precision_score(Y_Test, sgd_pred))    
    print('Recall: %.3f' % recall_score(Y_Test, sgd_pred))
    print('F1-Score: %.3f' % f1_score(Y_Test, sgd_pred))
    print('Accuracy: %.3f' % accuracy_score(Y_Test, sgd_pred))



if __name__ == "__main__":
    print("hi")
    try:
        algorithm_type = str(sys.argv[1])
        dataset_used   = str(sys.argv[2])       
        
        feature_extraction(algorithm_type,dataset_used)
    
    except Exception as e:
        print(e)
