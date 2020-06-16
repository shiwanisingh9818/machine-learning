'''objective/target----> To predict car year rescale value.'''

#import the libraries.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

##insert the data from external source...

data=pd.read_csv(r'C:\Users\HP\Desktop\Car_sales.csv')

'''data explore'''

data.head()

data.columns

'''data information

1.Manufacturer--->producer(objective type)
2.Model------->car model
3.Sales_in_thousands--->number of cars sale/continious
4.__year_resale_value--->target/cars sale in a year/continious
5.Vehicle_type---->car/pasenger-->important--->binominal
6.price_in_thousands-->car sales price in thousand/imp.--->continious
7.Engine-->engine size/imp.
8.Horsepower--->power of car/imp.
9.width
10.length
11.curb_weight
12.Fuel_capacity-->capacity of car to store the oil/imp.
13.Fuel_efficiency-->vehicle average/imp.
14.Latest_launch'--->cars launching date
15.Power_perf_factor--->power of engine

'''

'''Data explore'''
data.info()

data.isnull().sum()


'''empty values are

__year_resale_value    36
Price_in_thousands      2
Engine_size             1
Horsepower              1
Wheelbase               1
Width                   1
Length                  1
Curb_weight             2
Fuel_capacity           1
Fuel_efficiency         3
Power_perf_factor       2
'''

#treatment of null values

plt.hist('_year_rescale_value')

data['__year_resale_value'].fillna(data['__year_resale_value'].mean(),inplace=True)

plt.hist(data['Price_in_thousands'])

data['Price_in_thousands'].fillna(data['Price_in_thousands'].median(),inplace=True)

plt.hist('Engine_size')

data['Engine_size'].fillna(data['Engine_size'].mean(),inplace=True)

data.hist('Horsepower')
data['Horsepower'].fillna(data['Horsepower'].median(),inplace=True)

data.hist('Wheelbase')

data['Wheelbase'].fillna(data['Wheelbase'].median(),inplace=True)

data.hist('Width')
data['Width'].fillna(data['Width'].mean(),inplace=True)
data.hist('Length')
data['Length'].fillna(data['Length'].mean(),inplace=True)

data.hist('Curb_weight')
data['Curb_weight'].fillna(data['Curb_weight'].mean(),inplace=True)

data.hist('Fuel_capacity')
data['Fuel_capacity'].fillna(data['Fuel_capacity'].median(),inplace=True)

data.hist('Fuel_efficiency')
data['Fuel_efficiency'].fillna(data['Fuel_efficiency'].median(),inplace=True)

data.hist('Power_perf_factor')
data['Power_perf_factor'].fillna(data['Power_perf_factor'].median(),inplace=True)


data.isnull().sum()  #all values are filled


data.info()
'''sepreate objective and numerical'''
ob=[]
num=[]
for i in data:
    if(data[i].dtypes=='object'):
        ob.append(i)
    else:
        num.append(i)
        
print(ob)
print(num)

''' objective type of data are

['Manufacturer', 'Model', 'Vehicle_type', 'Latest_Launch']

'''

#treatment of these objective type of data

#1.Manufacture




from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['Manufacturer']=le.fit_transform(data['Manufacturer'])      

#2. Model

data['Model']=le.fit_transform(data['Model']) 

#3.Vehicle_type

data['Vehicle_type']=le.fit_transform(data['Vehicle_type'])

data.info()

###convert the data into date 



data['Latest_Launch']=pd.to_datetime(data['Latest_Launch'],infer_datetime_format=True)

#data['year']=data['Latest_Launch'].dt.year

#data['month']=data['Latest_Launch'].dt.month

#data['day']=data['Latest_Launch'].dt.day

#data['Latest_Launch'].min()

#data['Latest_Launch'].max()




#outlier treatment

sns.boxplot(data['Sales_in_thousands'])

data['Sales_in_thousands']=np.where(data['Sales_in_thousands']>200,200,data['Sales_in_thousands'])


sns.boxplot(data['Price_in_thousands'])

data['Price_in_thousands']=np.where(data['Price_in_thousands']>70,70,data['Price_in_thousands'])



sns.boxplot(data['Engine_size'])

data['Engine_size'] =np.where(data['Engine_size']>6,6,data['Engine_size'])


sns.boxplot(data['Horsepower'])
data['Horsepower']=np.where(data['Horsepower']>350,350,data['Horsepower'])

sns.boxplot(data['Wheelbase'])

data['Wheelbase']=np.where(data['Wheelbase']>135,135,data['Wheelbase'])

sns.boxplot(data['Width'])

sns.boxplot(data['Length'])

data['Length']=np.where(data['Length']>220,220,data['Length'])

sns.boxplot(data['Curb_weight'])

data['Curb_weight']=np.where(data['Curb_weight']>5.5,5.5,data['Curb_weight'])

sns.boxplot(data['Fuel_capacity'])

data['Fuel_capacity']=np.where(data['Fuel_capacity']>30,30,data['Fuel_capacity'])

sns.boxplot(data['Fuel_efficiency'])

sns.boxplot(data['Power_perf_factor'])

data['Power_perf_factor']=np.where(data['Power_perf_factor']>150,150,data['Power_perf_factor'])

data.drop(['Manufacturer','Model','Latest_Launch'],axis=1)


###########################################################################################

data.sort_values(["Latest_Launch"], axis=0, 
                 ascending=True, inplace=True)
data['index']=np.arange(0,157,1)
data
a=data.iloc[0:112,0:]
b=data.iloc[112:,0:]
b
a.pop('Latest_Launch')
b.pop('Latest_Launch')
a.pop('index')
b.pop('index')
type(a)

#a.drop(['year','month','day'],axis=1)
a.columns

########remove the unneccesary data



#data cleaning is completed

y=a.__year_resale_value

x=a.drop(['__year_resale_value'],axis=1)

y1=b.__year_resale_value

x1=b.drop(['__year_resale_value'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.003,random_state=42)


x_train.head()

x_train.shape
y_train.shape
x_test.shape







from sklearn.linear_model import LinearRegression

model=LinearRegression()

##fit the data to model
x_train.columns

model.fit(x_train,y_train)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.95,random_state=42)


y_pred=model.predict(x_test)

model.score(x,y)
'''0.7816425136541294'''











































columns = data.columns.tolist()


for i in columns:
    a[i] = a[i].transform(a[i].fillna('NA'))
    features = a[a.columns.difference(['Target'])]
    features = features.fillna(0)
    features.info()