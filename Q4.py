#CS464_HW1_DorraElMekki
import csv
import math
import sys
import os


V = 26507; #Vocabulary size

print("getting the dataset for training")

labels = open('question-4-train-labels.csv',"r+")
csvFile = csv.reader(labels,delimiter=',')
labelsDatasetTrain = list(csvFile)

feature = open('question-4-train-features.csv',"r+")
csvFile= csv.reader(feature,delimiter=',')
featuresDatasetTrain = list(csvFile)

print("start trainig the model")

#parameters
N = 0;#Number of emails
N0 = 0;#Number of medical emails
N1 = 0;#Number of space emails

T0 = [0.]*V;#occurrences of the words in medical,we use float because we are going to divide numbers 
T1 = [0.]*V;#occurrences of the words in space emails


SUM0 = 0.0;#sum(words)in all medical emails
SUM1 = 0.0;#sum(words) in all space emails

for i in range(len(labelsDatasetTrain)):
    if(labelsDatasetTrain[i][0]=='0'):
        N0 = N0 + 1;
        for j in range(0,V-1):
           T0[j] = T0[j] + float(featuresDatasetTrain[i][j]);
           SUM0 = SUM0 + float(featuresDatasetTrain[i][j])                         
              
    if(labelsDatasetTrain[i][0]=='1'):
            N1 = N1 + 1;
            for j in range(0,V-1):
               T1[j] = T1[j] + float(featuresDatasetTrain[i][j]);
               SUM1 = SUM1 + float(featuresDatasetTrain[i][j])              
    N=N+1;
    

P0 = float(N0)/N;#probability of medical: class 0 
P1 = float(N1)/N;#probability of space: class 0 



for j in range(0,V-1):
    T0[j] = float(T0[j])/SUM0;#prob of words j in medical class 0=medical
    T1[j] = float(T1[j])/SUM1;#prob of words j in space class 1=space
print("finish trainig")

##################################################################################

print("start testing")

print("getting the dataset for testing")

labels = open('question-4-test-labels.csv',"r+")
csvFile = csv.reader(labels, delimiter=',')
labelsDatasetTest = list(csvFile)

N0Test=0
N1Test=0

testLabes = []
for i in range(len(labelsDatasetTest)):
    if (labelsDatasetTest[i][0]=='0'):
        testLabes.append(0)
        N0Test = N0Test +1
    if(labelsDatasetTest[i][0]=='1'):
        testLabes.append(1)
        N1Test = N1Test +1

features = open('question-4-test-features.csv',"r+")
csvFile = csv.reader(features,delimiter=',')
featuresDatasetTest = list(csvFile)


testSize=len(featuresDatasetTest)
PT0=[0.]*testSize; #Bayes Rule 
PT1=[0.]*testSize;


Prediction = [];    
for i in range(len(featuresDatasetTest)):
    for j in range(0,V-1):
        if (T0[j] != 0):
            PT0[i] = math.log(T0[j]) * float(featuresDatasetTest[i][j]) + PT0[i] ;
        if (T1[j] != 0):
            PT1[i] = math.log(T1[j]) * float(featuresDatasetTest[i][j]) + PT1[i] ; 
    PT0[i] = math.log(P0) + PT0[i] 
    PT1[i] = math.log(P1) + PT1[i]

    #we compare the two values to know which one maximize the value of the prob, to take decision for the class
    if (PT0[i] > PT1[i]) :
        Prediction.append(0)
    if (PT1[i] >= PT0[i]) :
        Prediction.append(1)

TN= 0; #predicting 1 and the real value is 1      
TP= 0; #predicting 1 and the real value is 1   
for i in range(len(featuresDatasetTest)):
    if (Prediction[i] == testLabes[i]) and (testLabes[i] == 0):
            TN=TN+1
    if (Prediction[i] == testLabes[i]) and (testLabes[i] == 1):
            TP=TP+1
            
CorrectPrediction = TP + TN;

FP = N0Test - TN;
MD = N1Test - TP;

accuracy = float(CorrectPrediction) / len(featuresDatasetTest);
print("test set accuracy is :")
print(accuracy)


report = open('Result_Q4.txt', 'w')


report.write("MLE estimator\n")
report.write("\n" )
report.write("The testing accuracy:")
report.write(str(accuracy))
report.write("\n" )
report.write("Wrong predictions:\n")
report.write("prediction medical , the real value is space :" )
report.write(str (FP))
report.write("\n" )
report.write("prediction space , the real value is medical :" )
report.write(str(MD))
report.write("\n" )
report.write("total false predictions: " )
report.write(str(MD+FP))
report.write("\n" )
report.write("Size of the testing data: " )
report.write(str(testSize))
report.write("\n" )
report.close()

    
