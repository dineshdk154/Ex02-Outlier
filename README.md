# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them

## EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Get the information about the data

### STEP 3
Detect the Outliers using IQR method and Z score

### STEP 4
Remove the outliers

### STEP 5
Plot the datas using Box Plot

## CODE
(1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe.
```
import pandas as pd import numpy as np import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 - Data Science/bhp.csv") df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df) q1 = df['price_per_sqft'].quantile(0.25) q3 = df['price_Aper_sqft'].quantile(0.75) print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1 ul = q3+1.5IQR ll = q1-1.5IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))] df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)`

(3) Examine price_per_sqft column and use zscore of 3 to remove outliers. from scipy import stats
z = np.abs(stats.zscore(df['price_per_sqft'])) df2 = df[(z<3)] df2

print(df2.shape) sns.boxplot(x="price_per_sqft",data=df2)

(4)(i) For the data set height_weight.csv detect weight outliers using IQR method df3 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 - Data Science/height_weight.csv") df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25) q3 = df3['weight'].quantile(0.75) print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1 ul = q3+1.5IQR ll = q1-1.5IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))] df4

df4.shape

<img width="276" alt="2 10" src="https://user-images.githubusercontent.com/93427345/191421710-aad50409-e824-4ad5-8c87-665de18fdbbb.png">
sns.boxplot(x="weight",data=df4) (4)(ii) For the data set height_weight.csv detect height outliers using IQR method sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25) q3 = df3['height'].quantile(0.75) print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1 ul = q3+1.5IQR ll = q1-1.5IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))] df5

df5.shape

sns.boxplot(x="height",data=df5)
```
## OUTPUT
(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
### Dataset
<img width="448" alt="2 1" src="https://user-images.githubusercontent.com/93427345/191420861-264b11f4-460e-4f1d-a3e3-e427b5a47232.png">
Dataset head
<img width="455" alt="2 2" src="https://user-images.githubusercontent.com/93427345/191421018-00152688-4928-4089-a82c-b2d2e188e15b.png">
Dataset info 
<img width="272" alt="2 3" src="https://user-images.githubusercontent.com/93427345/191421159-ef2bfc07-cdd1-45b0-a979-1d54a682eb78.png">
Data describe
<img width="450" alt="2 4" src="https://user-images.githubusercontent.com/93427345/191421241-96444714-e91d-492b-8d13-fcd1f88a36d7.png">
Null values
<img width="125" alt="2 5" src="https://user-images.githubusercontent.com/93427345/191421264-375d009f-fd61-437e-bd76-3590dea94c6e.png">
Dataset shape
<img width="63" alt="2 6" src="https://user-images.githubusercontent.com/93427345/191421323-ca932863-bb30-4480-87df-03b12ea23bbf.png">
Box plot of price_per_sqft column with outliers
<img width="328" alt="2 7" src="https://user-images.githubusercontent.com/93427345/191421383-62a7abb9-8d67-4a1c-a81f-26541c8ca384.png">
price_per_sqft - Dataset after removing outliers
<img width="455" alt="2 8" src="https://user-images.githubusercontent.com/93427345/191421579-8d6f9cb2-982c-4702-b77f-0a99dcbcc20a.png">
price_per_sqft - Shape of Dataset after removing outliers
<img width="67" alt="2 9" src="https://user-images.githubusercontent.com/93427345/191421667-77bba9b6-7ec6-45d7-be34-95a4142a8673.png">
Box Plot of price_per_sqft column without outliers
<img width="276" alt="2 10" src="https://user-images.githubusercontent.com/93427345/191421744-1549014b-a5bb-4e86-bb9f-fe64a64d53ee.png">
Examine price_per_sqft column and use zscore of 3 to remove outliers. Dataset after removal of outlier using z score
<img width="457" alt="2 11" src="https://user-images.githubusercontent.com/93427345/191421869-0d31589a-3184-48d5-a472-97246951beb8.png">
Shape of Dataset after removal of outlier using z score
<img width="70" alt="2 12" src="https://user-images.githubusercontent.com/93427345/191422311-375cea5d-41b6-4b90-8c5c-70bd3cfdadd8.png">
price_per_sqft column after removing outliers
<img width="320" alt="2 13" src="https://user-images.githubusercontent.com/93427345/191422633-e770efde-e676-4e75-953c-e1b87d091b78.png">
(4) For the data set height_weight.csv detect weight and height outliers using IQR method Dataset
<img width="211" alt="2 14" src="https://user-images.githubusercontent.com/93427345/191422770-f962f8eb-9769-4b0b-a5c4-4fd77597819a.png">
Datahead
<img width="191" alt="2 15" src="https://user-images.githubusercontent.com/93427345/191422798-767c36b2-f30d-4a8a-9427-0c0d5391a07d.png">
Dataset info
<img width="221" alt="2 16" src="https://user-images.githubusercontent.com/93427345/191422826-d2a9429a-7bf2-4ef4-b475-ea4fdeec1730.png">
Dataset describe
<img width="197" alt="2 17" src="https://user-images.githubusercontent.com/93427345/191422871-79b91c53-8cc1-44a9-b51c-abd946dabed8.png">
Null values
<img width="80" alt="2 18" src="https://user-images.githubusercontent.com/93427345/191422969-50b67e52-a9ce-44bb-afab-3f47e5019205.png">
Dataset Shape
<img width="70" alt="2 19" src="https://user-images.githubusercontent.com/93427345/191423031-c4624ce9-45f3-4bea-9c03-6d54215e7727.png">
Weight - With outlier
<img width="273" alt="2 20" src="https://user-images.githubusercontent.com/93427345/191423096-9f98098c-5528-48ee-a721-49e48b3dfdd9.png">
Weight - Dataset after removing Outliers using IQR method
<img width="221" alt="2 21" src="https://user-images.githubusercontent.com/93427345/191423252-d3ebd13c-febb-45d0-800e-72540988cbae.png">
Weight - Shape of Dataset after removing Outliers using IQR method
<img width="65" alt="2 22" src="https://user-images.githubusercontent.com/93427345/191423312-0cd6f2b5-5c17-4a5d-95dc-82fc1d0355a2.png">
Weight - Without Outliers using IQR method
<img width="341" alt="2 23" src="https://user-images.githubusercontent.com/93427345/191423368-f044608b-8efb-4941-a9ce-3d3b2d2cc8c2.png">
Height - With outliers
<img width="282" alt="2 24" src="https://user-images.githubusercontent.com/93427345/191423597-6012d3a3-3390-4f22-b0e4-c70b9d95a01a.png">
Height - Dataset after removing Outliers using IQR method
<img width="210" alt="2 25" src="https://user-images.githubusercontent.com/93427345/191423661-a0dfede9-4724-4423-b356-7725e3b97ee4.png">
Height - Shape of Dataset after removing Outliers using IQR method
<img width="80" alt="2 26" src="https://user-images.githubusercontent.com/93427345/191423717-6746d208-5953-4c0f-a371-4658c5e0566e.png">
Height - Without Outliers using IQR method
<img width="341" alt="2 27" src="https://user-images.githubusercontent.com/93427345/191423849-4639f859-0a7a-45c6-9ff0-28b6c8557a65.png">

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.



