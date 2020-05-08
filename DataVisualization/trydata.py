# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:09:56 2020

@author: emmav
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from pylab import rcParams

import scipy
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


mydata = pd.read_csv('san-mateo-county-2018.csv')
df = DataFrame(mydata, columns = ['Employee Name','Job Title','Base Pay','Overtime Pay','Benefits','Total Pay',
                            'Pension Debt','Total Pay & Benefits','Year','Notes','Agency','Status'])
salaries = pd.read_csv('title-and-salaries.csv')
altsalaries = pd.read_csv('title-and-salaries.csv')

#normal distribution
def plotTotalPay (data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1,n+1)/n
    return x,y
x,y = plotTotalPay(df['Total Pay'])


plt.figure(1,figsize = (8,7))
sns.set()
plt.plot(x,y,marker=".",linestyle="none")
plt.xlabel("Total Pay")
plt.ylabel("Distribution")

sample_set = np.random.normal(np.mean(df["Total Pay"]), np.std(df["Total Pay"]),size = 1000)
x_check, y_check = plotTotalPay(sample_set)
plt.plot(x_check,y_check)
plt.legend(('Normal Distribution','Our Data Set'),loc = 'lower right')
plt.title("Normal Distribution Check")

#Adult Psychiatrist stats

def salariesA (salaries):
    filter = salaries["Job Title"] == "Adult Psychiatrist"
    salaries.where(filter, inplace = True)
    salaries = salaries.dropna()
    df = pd.DataFrame(salaries, columns = ['Job Title','Total Pay'])
    adult_mean = df['Total Pay'].mean()
    adult_std = df['Total Pay'].std()
    n = len(df.index)
    return adult_mean, adult_std, n



#Sheriff's Lieutenant stats
def salariesB (salaries):
    filter = salaries["Job Title"] == "Sheriff's Lieutenant"
    salaries.where(filter, inplace = True)
    salaries=salaries.dropna()
    df = pd.DataFrame(salaries,columns = ['Job Title','Total Pay'])
    sheriff_mean = df['Total Pay'].mean()
    sheriff_std = df['Total Pay'].std()
    n = len(df.index)
    return sheriff_mean, sheriff_std, n

#hypothesis test
outputA = [salariesA(salaries)]
outputB = [salariesB(altsalaries)]
outputA = np.reshape(outputA,(1,3))
outputB = np.reshape(outputB, (1,3))

mean1= outputA.item(0)
mean2= outputB.item(0)
st1= outputA.item(1)
st2= outputB.item(1)
n1= outputA.item(2)
n2= outputB.item(2)

print("Hypothesis test: The salaries of Adult Psychatrists are different than the salaries of Sheriff Lieutenants.")
print("H0: m1 = m2")
print("H1: m1 != m2")
ttest,pval = stats.ttest_ind_from_stats(mean1,st1,n1,mean2,st2,n2)
print("P-value = ", pval)

if pval < .05:
    print("We reject the null hypothesis")
else:
    print("We fail to reject the null hypothesis")

#Correlation Coefficent
BP = mydata['Base Pay']
OP = mydata['Overtime Pay']
TPB = mydata['Total Pay & Benefits']

pearsonr_coefficient, p_value = pearsonr(BP, OP)
print('PearsonR Correltaion Coefficient %0.3f' % (pearsonr_coefficient))

pearsonr_coefficient, p_value = pearsonr(TPB, OP)
print('PearsonR Correltaion Coefficient %0.3f' % (pearsonr_coefficient))


#correlation graph BP vs OP
plt.figure(2)
plt.style.use('ggplot')
plt.scatter(BP,OP)
plt.title("Base Pay vs. Overtime Pay")
plt.xlabel("Base Pay")
plt.ylabel("Overtime Pay")
plt.show()

#correlation graph TPB vs OP
plt.figure(3)
plt.style.use('ggplot')
plt.scatter(TPB,OP)
plt.title("Total Pay & Benefits vs. Overtime Pay")
plt.xlabel("Total Pay & Benefits")
plt.ylabel("Overtime Pay")














