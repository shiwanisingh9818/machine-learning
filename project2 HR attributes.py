'''import the library'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''import the data from external source'''

data=pd.read_csv(r'C:\Users\Public\Documents\ibm hr.csv')

'''data exploration'''

data.head()

data.info()

'''feature selection'''

sns.catplot(x='Age',hue='Attrition',kind='count',data=data)


sns.violinplot(x='Age',hue='Attrition',data=data)

#important
sns.catplot(x='BusinessTravel',hue='Attrition',kind='count',data=data)

'''data says
1.Travel_Rarely----->high chance to left the job

so it is important
'''
sns.violinplot(x='DailyRate',y='Age',hue='Attrition',data=data)

#it is also impotant

sns.catplot(x='Department',hue='Attrition',kind='count',data=data)
# Research and developments employees mostly leave their jobs

sns.catplot(x='EducationField',hue='Attrition',kind='count',data=data) 
#important

sns.violinplot(x='Attrition',y='DistanceFromHome',data=data)
#important because minimum destance employes risk is very less to left the job

sns.catplot(x='Education',hue='Attrition',kind='count',data=data) 

#3 labels job left high

data['EmployeeCount'].value_counts()

###all rows are 1 so we can left it.....it does not create any impact

data['EmployeeNumber'].value_counts()

sns.violinplot(x='Attrition',y='EmployeeNumber',data=data)

##it is important

sns.catplot(x='EnvironmentSatisfaction',hue='Attrition',kind='count',data=data) 

#it is important bcz 3 and 4 labels has high chance to not leave the job

sns.violinplot(x='Gender',y='Age',hue='Attrition',data=data)

##important

sns.violinplot(x='Gender',y='HourlyRate',hue='Attrition',data=data)

#important


sns.catplot(x='JobSatisfaction',hue='Attrition',kind='count',data=data) 

#important


sns.catplot(x='MaritalStatus',hue='Attrition',kind='count',data=data) 

'''
single------take risk
married------less chance to take risk
divorced-----to less chance to take risk
'''
sns.violinplot(x='Attrition',y='MonthlyIncome',data=data)

##income is too important bcz less than 5000 peoples left their job


sns.catplot(x='NumCompaniesWorked',hue='Attrition',kind='count',data=data) 

###those work 1 company left their job mostly


sns.catplot(x='JobLevel',hue='Attrition',kind='count',data=data) 
#important

sns.catplot(x='JobRole',hue='Attrition',kind='count',data=data) 
#important

data['Over18'].value_counts()
sns.violinplot(x='Attrition',y='Over18',data=data)

#all employes are over 18 in ibm so we can drop this columns
data['OverTime'].value_counts()
sns.catplot(x='OverTime',hue='Attrition',data=data,kind='count')

##those peoples are working in overtime they are left their job as compare to no

sns.violinplot(x='Attrition',y='PercentSalaryHike',data=data)

#important

sns.violinplot(x='Attrition',y='PerformanceRating',data=data)

#important  it seems nuteral but having littile variation


sns.catplot(x='RelationshipSatisfaction',hue='Attrition',data=data,kind='count')

#label 3 peoples are more satisfied


data['StandardHours'].value_counts()

##standard hours are 80 so it is not relevent

data['StockOptionLevel'].value_counts()

sns.catplot(x='StockOptionLevel',hue='Attrition',data=data,kind='count')

data['TotalWorkingYears'].value_counts()

sns.violinplot(x='Attrition',y='TotalWorkingYears',data=data)

#important

data['TrainingTimesLastYear'].value_counts()

sns.catplot(x='TrainingTimesLastYear',hue='Attrition',data=data,kind='count')

##thoes are doing 2 times training they are left their job

data.info()

sns.violinplot(x='Attrition',y='WorkLifeBalance',data=data)

sns.violinplot(x='Attrition',y='YearsAtCompany',data=data)

sns.violinplot(x='Attrition',y='YearsInCurrentRole',data=data)


sns.violinplot(x='Attrition',y='YearsSinceLastPromotion',data=data)

sns.violinplot(x='Attrition',y='YearsWithCurrManager',data=data)


'''data cleaning part'''

data.isnull().sum()

'''remove the unrelevent data'''
data=data.drop(['EmployeeCount','Over18','StandardHours'],axis=1)


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data['Attrition']=le.fit_transform(data['Attrition'])

data['BusinessTravel']=le.fit_transform(data['BusinessTravel'])

data['Department']=le.fit_transform(data['Department'])

data['EducationField']=le.fit_transform(data['EducationField'])

data['JobRole']=le.fit_transform(data['JobRole'])

data['Gender']=le.fit_transform(data['Gender'])

data['MaritalStatus']=le.fit_transform(data['MaritalStatus'])


data['OverTime']=le.fit_transform(data['OverTime'])


data.info()

y=data['Attrition']

x=data.drop(['Attrition'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

##logistic Regression

from sklearn.linear_model import LogisticRegression

model1=LogisticRegression()

model1.fit(x,y)

y_pred=model1.predict(x_test)

model1.score(x,y)

#0.8734693877551021


#knn


from sklearn.neighbors import KNeighborsClassifier

model2=KNeighborsClassifier(metric='euclidean')

model2.fit(x,y)

y_pred=model2.predict(x_test)

model2.score(x,y)

#0.854421768707483

##naive bays

from sklearn.naive_bayes import GaussianNB

model3=GaussianNB()

model3.fit(x,y)

model3.score(x,y)
#0.8


##svm

from sklearn.svm import SVC

model4=SVC()

model4.fit(x,y)

model4.score(x,y)

#1.0


#Decision Tree

from sklearn.tree import DecisionTreeClassifier

model4=DecisionTreeClassifier()

model4.fit(x,y)

model4.score(x,y)

#1.0

from sklearn.tree import export_graphviz

eg=export_graphviz(model4,out_file='dt.txt')




#Ensemble Technique





from sklearn.linear_model import LogisticRegression

le=LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier

knc=KNeighborsClassifier(metric='euclidean')

from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

from sklearn.svm import SVC

sv=SVC()

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()


##voting

from sklearn.ensemble import VotingClassifier

algo=[('LogisticRegression',le),('KNeighborsClassifier',knc),('GaussianNB',gb),('SVC',sv),('DecisionTreeClassifier',dtc)]

vot=VotingClassifier(algo)

vot.fit(x,y)

vot.score(x,y)

#0.9510204081632653


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x,y)

rf.score(x,y)

#0.9836734693877551


'''conclussion---->we using different classification algorithem in which model is 
overfited using svm and decision tree'''



