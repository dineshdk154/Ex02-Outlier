# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
    
# Explanation :
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# Algorithm:

STEP 1
Read the given Data.

STEP 2
Get the information about the data.

STEP 3
Detect the Outliers using IQR method and Z score.

STEP 4
Remove the outliers.

STEP 5
Plot the datas using Box Plot.
 
 
# Program:
Developed by: DINESH KUMAR M

Register number:212221220011
```
# 1
import pandas as ps
import numpy as np
import seaborn as sns
df=ps.read_csv("bhp.csv")
df
df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape
sns.boxplot(x="price_per_sqft",data=df)

# 2
q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)
IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR
df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1
df1.shape
sns.boxplot(x='price_per_sqft',data=df1)

# 3
from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2
print(df2.shape)
sns.boxplot(x='price_per_sqft',data=df2)

# 4 (i) 
df3=ps.read_csv('height_weight.csv')
df3
df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape
sns.boxplot(x='weight',data=df3)

# 4 (ii)
q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)
IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR
df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4.shape
sns.boxplot(x='height',data=df4)
```
# Output:

Data for bhp.csv

![image](https://user-images.githubusercontent.com/128135186/230089239-3f2644ce-9784-42f2-8101-fb0355e2613d.png)

Dataset head

![image](https://user-images.githubusercontent.com/128135186/230089606-22c33f96-4ed7-4de2-aace-a8a19cd9f49e.png)

Dataset describe

![image](https://user-images.githubusercontent.com/128135186/230089736-bbdce6c9-538f-4d7d-bb4b-29aec8be3862.png)

Dataset info

![image](https://user-images.githubusercontent.com/128135186/230089825-9bdd758a-c0d3-41c1-83d3-e851d66465b5.png)

Null values

![image](https://user-images.githubusercontent.com/128135186/230089893-206b5997-8bbe-4645-9307-8bf731317e8c.png)

Dataset shape with outliers

![image](https://user-images.githubusercontent.com/128135186/230090044-6c9258e8-8797-4af8-b967-3c2de3265cd5.png)

Dataset boxplot with outliers

![image](https://user-images.githubusercontent.com/128135186/230090313-82050150-9233-4aba-82ec-93cd1640541b.png)

Dataset without outliers

![image](https://user-images.githubusercontent.com/128135186/230090464-50ae920c-3e8c-4e37-8941-1f4f247b2548.png)

![image](https://user-images.githubusercontent.com/128135186/230090517-fc89a841-89b5-4a88-babe-3fb29826b1a6.png)


Dataset shape without outliers

![image](https://user-images.githubusercontent.com/128135186/230091406-aa380e4d-2289-4b93-bb70-1726388252ef.png)


Dataset boxplot without outliers 

![image](https://user-images.githubusercontent.com/128135186/230091907-7a5893d1-c7bf-4355-afe0-6e63e148bbd0.png)

Dataset after removal of outliers using Z-score

![image](https://user-images.githubusercontent.com/128135186/230092141-fc577582-e9dc-4dcf-a0a4-713aabf4e741.png)

Dataset shape after removal outliers

![image](https://user-images.githubusercontent.com/128135186/230093707-e48dceae-1882-466a-a3c6-f74071845e2a.png)

Boxplot after removal of outliers

![image](https://user-images.githubusercontent.com/128135186/230093830-f21cfdb0-f37e-402d-b4a7-d3034909c2b4.png)

Dataset of height_weight.csv

![image](https://user-images.githubusercontent.com/128135186/230093965-4a0be2a5-e7be-44eb-b2ee-a4bc8257ae08.png)

Dataset head

![image](https://user-images.githubusercontent.com/128135186/230094034-9994248a-7457-4822-b13a-3a39558e6203.png)

Dataset info

![image](https://user-images.githubusercontent.com/128135186/230094093-3e106965-2628-4ad5-9c89-96a55bde46b7.png)

Dataset describe

![image](https://user-images.githubusercontent.com/128135186/230094161-b04899cd-86bb-45ba-b842-5b469320eadc.png)

Null values

![image](https://user-images.githubusercontent.com/128135186/230094212-b349c14f-6374-42e9-964d-0f981ed801cb.png)

Boxplot with outliers

![image](https://user-images.githubusercontent.com/128135186/230094279-146c82d3-1e0a-432f-b081-4e444b2f6e49.png)


Dataset after removal of outliers using IQR method

![image](https://user-images.githubusercontent.com/128135186/230094334-f45835c7-a432-4b1d-9b5a-b054c090ff77.png)

![image](https://user-images.githubusercontent.com/128135186/230094427-5a83c9f8-affc-4446-b322-62d33f8af8db.png)

Dataset shape

![image](https://user-images.githubusercontent.com/128135186/230094545-b0d7ab03-9a83-4e11-8285-adf43a9be59f.png)

Boxplot after removal of outliers using IQR method

![image](https://user-images.githubusercontent.com/128135186/230094595-bb3a6b50-c06a-4683-949f-5d6922c8fda7.png)








