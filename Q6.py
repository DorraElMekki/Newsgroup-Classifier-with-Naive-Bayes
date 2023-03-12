#CS464_HW1_DorraElMekki
import csv
import math
import sys
import os


V = 26507
print("getting dataset")
labels = open('question-4-train-labels.csv',"r+")
csvFile = csv.reader(labels,delimiter=',')
labelsDatasetTrain = list(csvFile)

feature = open('question-4-train-features.csv',"r+")
csvFile= csv.reader(feature,delimiter=',')
featuresDatasetTrain = list(csvFile)


N00 = [0.]*V;
N10 = [0.]*V;
N11 = [0.]*V;
N01 = [0.]*V;
N = [0.]*V;
print("calculating MutualInf")

MutualInf=[0.]*V;
for i in range(len(labelsDatasetTrain)):
    if (labelsDatasetTrain[i][0]=='0'):
        for j in range(V-1):
            if(int(featuresDatasetTrain[i][j]) != 0):               
                 N10[j] = N10[j] + 1 
            if(int(featuresDatasetTrain[i][j]) == 0):               
                 N00[j] = N00[j] + 1

    if (labelsDatasetTrain[i][0]=='1'):
        for j in range(V-1):
            if(int(featuresDatasetTrain[i][j]) != 0):               
                 N11[j] = N11[j] + 1 
            if(int(featuresDatasetTrain[i][j]) == 0):               
                 N01[j] = N01[j] + 1
for j in range(V-1):
    N[j] = N11[j] + N10[j] + N00[j] + N01[j]

for j in range(V-1):
    if (N11[j] + N10[j] != 0)and(N11[j] + N01[j] != 0)and(N00[j] + N01[j]!= 0)and(N00[j] + N10[j] != 0):
        m1=(N[j]*N11[j])/((N11[j]+N10[j])*(N11[j]+N01[j]))
        m2=(N[j]*N01[j])/((N00[j]+N01[j])*(N11[j]+N01[j]))
        m3=(N[j]*N10[j])/((N11[j]+N10[j])*(N00[j]+N10[j]))
        m4=(N[j]*N00[j])/((N00[j]+N01[j])*(N00[j]+N10[j]))

        MutualInf[j]=0
        if(m1 != 0):
            MutualInf[j] = ( N11[j]/ N[j] ) * math.log(m1,2)
        if (m2 != 0):
            MutualInf[j] += ( N01[j]/N[j]) * math.log(m2, 2)
        if (m3 != 0):
            MutualInf[j] += ( N10[j]/N[j]) * math.log(m3, 2)
        if (m4 != 0):
            MutualInf[j] += ( N00[j]/N[j]) * math.log(m4, 2)


sortedMutualInf = [0.]*V
for i in range(V-1):
    sortedMutualInf[i] = MutualInf[i]

indice = [0.]*V;
for i in range(V-1):
    indice[i] = i

print("Sorting features by calculating mutual information")
for i in range(V-1):
    posmax=i
    indice[i] = i
    for j in range(i+1,V-1):
        if sortedMutualInf[j]>sortedMutualInf[posmax]:
            posmax=j
    (indice[i],indice[posmax])=(indice[posmax],indice[i])           
    (sortedMutualInf[i],sortedMutualInf[posmax])=(sortedMutualInf[posmax],sortedMutualInf[i])

print("write result in Result_Q6.txt")

with open('Result_Q6.txt', 'w') as result:
    result.write("Mutual information\n")
    result.write("Indices and mutual information scores of the 10 features in descending order\n")
    for i in range(10):
        result.write("Feature N: ")
        result.write(str (indice[i]))
        result.write("  =>mutual information equal to: ")
        result.write(str (sortedMutualInf[i]))
        result.write("\n")

