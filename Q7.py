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

print("start trainig the model...")

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
    T0[j] = (float(T0[j])+1)/(SUM0+V);#prob of words j in medical class 0=medical
    T1[j] = (float(T1[j])+1)/(SUM1+V);#prob of words j in space class 1=space
print("finish trainig")


#########################  Mutual Information
N00 = [0.]*V;
N10 = [0.]*V;
N11 = [0.]*V;
N01 = [0.]*V;
N = [0.]*V;
print("calculating MutualInf")

MutualInf=[0.]*V;
for i in range(len(labelsDatasetTrain)):
    if(labelsDatasetTrain[i][0]=='0'):
        for j in range(0,V-1):
            if(int(featuresDatasetTrain[i][j]) != 0):               
                 N10[j] = N10[j] + 1 
            if(int(featuresDatasetTrain[i][j]) == 0):               
                 N00[j] = N00[j] + 1
                 
    if(labelsDatasetTrain[i][0]=='1'):
        for j in range(0,V-1):
            if(int(featuresDatasetTrain[i][j]) != 0):               
                 N11[j] = N11[j] + 1 
            if(int(featuresDatasetTrain[i][j]) == 0):               
                 N01[j] = N01[j] + 1          
for j in range(0,V-1):
    N[j] = N11[j] + N10[j] + N00[j] + N01[j]

for j in range(0,V-1):
    if (N11[j] + N10[j] != 0)and(N11[j] + N01[j] != 0)and(N00[j] + N01[j]!= 0)and(N00[j] + N10[j] != 0):
        m1=(N[j]*N11[j])/((N11[j]+N10[j])*(N11[j]+N01[j]))
        m2=(N[j]*N01[j])/((N00[j]+N01[j])*(N11[j]+N01[j]))
        m3=(N[j]*N10[j])/((N11[j]+N10[j])*(N00[j]+N10[j]))
        m4=(N[j]*N00[j])/((N00[j]+N01[j])*(N00[j]+N10[j]))
        
        MutualInf[j]=0
        if(m1 != 0):
            MutualInf[j] = ( N11[j]/ N[j] ) * math.log(m1,2)
        if(m2 != 0):
            MutualInf[j] = MutualInf[j] + ( N01[j]/N[j]) * math.log(m2, 2)
        if(m3 != 0):
            MutualInf[j] = MutualInf[j] + ( N10[j]/N[j]) * math.log(m3, 2)
        if(m4 != 0):
            MutualInf[j] = MutualInf[j] + ( N00[j]/N[j]) * math.log(m4, 2)

   
sortedMutualInf = [0.]*V
for i in range(0,V-1):
    sortedMutualInf[i] = MutualInf[i]

indice = [0.]*V;
for i in range(0,V-1):
    indice[i] = i

print("Sorting features by calculating mutual information")
for i in range(0,V-1):
    posmax=i
    indice[i] = i
    for j in range(i+1,V-1):
        if sortedMutualInf[j]>sortedMutualInf[posmax]:
            posmax=j
    (indice[i],indice[posmax])=(indice[posmax],indice[i])           
    (sortedMutualInf[i],sortedMutualInf[posmax])=(sortedMutualInf[posmax],sortedMutualInf[i])
  
############################################### testing 

print("start testing...")

print("getting the dataset for testing")

labels = open('question-4-test-labels.csv',"r+")
csvFile = csv.reader(labels, delimiter=',')
labelsDatasetTest = list(csvFile)



N0Test=0
N1Test=0

testLabels = []
for i in range(len(labelsDatasetTest)):
    if (labelsDatasetTest[i][0]=='0'):
        testLabels.append(0)
        N0Test = N0Test +1
    if(labelsDatasetTest[i][0]=='1'):
        testLabels.append(1)
        N1Test = N1Test +1

features = open('question-4-test-features.csv',"r+")
csvFile = csv.reader(features,delimiter=',')
featuresDatasetTest = list(csvFile)


testSize=len(featuresDatasetTest)




accuracy=[.0]*V
iteration=0
while(V>1):

 ######### we should calculate new values devided by the new sum of values
 for j in range(0,V-1):
     T0[j] = (float(T0[indice[j]])+1)/(SUM0+V);#prob of words j in medical class 0=medical
     T1[j] = (float(T1[indice[j]])+1)/(SUM1+V);#prob of words j in space class 1=space
 
 #########
 PT0=[0.]*testSize; #Bayes Rule 
 PT1=[0.]*testSize;
 Prediction = [];    
 for i in range(len(featuresDatasetTest)):
     for j in range(0,V-1):
         if (T0[j] != 0):
             PT0[i] = math.log(T0[indice[j]]) * float(featuresDatasetTest[i][indice[indice[j]]]) + PT0[i] ;
         if (T1[j] != 0):
             PT1[i] = math.log(T1[indice[j]]) * float(featuresDatasetTest[i][indice[indice[j]]]) + PT1[i] ; 
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
     if (Prediction[i] == testLabels[i]) and (testLabels[i] == 0):
             TN=TN+1
     if (Prediction[i] == testLabels[i]) and (testLabels[i] == 1):
             TP=TP+1
            
 CorrectPrediction = TP + TN;

 FP = N0Test - TN;
 MD = N1Test - TP;

 accuracy[iteration] = float(CorrectPrediction) / len(featuresDatasetTest);
 V=V-1
 iteration=iteration+1

print("the maximum of test set accuracy number is:")
print(max(accuracy))
#plt.plot(accuracy)


print("write result in Result_Q7.txt")
result = open('Result_Q7.txt', 'w')
result.write("the maximum of test set accuracy number is:")
result.write(str(max(accuracy)))
result.close();

