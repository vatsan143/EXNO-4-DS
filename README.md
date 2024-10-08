# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
# PROGRAM:
```
###NAME : SRIVATSAN G
###REG NO : 212223230216
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
missing

data2=data.dropna(axis=0)
data2
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
data2
new_data=pd.get_dummies(data2, drop_first=True)
new_data

columns_list=list(new_data.columns)
print(columns_list)
features=list(set(columns_list)-set(['SalStat']))
print(features)
y=new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
data.shape
###NAME : SRIVATSAN G
###REG NO : 212223230216
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
###NAME : SRIVATSAN G
###REG NO : 212223230216
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
tips.time.unique()

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
# OUTPUT:
![image](https://github.com/user-attachments/assets/ac09f4cb-9cc4-4199-bb34-5a27ed2056af)

![image](https://github.com/user-attachments/assets/2d47b30c-9725-47e2-a7b5-70ef25ca34dd)

![image](https://github.com/user-attachments/assets/d221e18a-78bd-40a4-a74c-131a46d67fb1)

![image](https://github.com/user-attachments/assets/005ff60d-9610-4732-ae62-4706b5382b2c)

![image](https://github.com/user-attachments/assets/77ffd7ff-3bef-4744-9419-1b52d1b0a2f6)

![image](https://github.com/user-attachments/assets/8daf0067-2ff7-46d7-98ef-69db91cce1ae)

![image](https://github.com/user-attachments/assets/e7bde823-d91f-4f3a-8dc1-20e37f02a76b)

![image](https://github.com/user-attachments/assets/d4851716-8b57-4bc5-8a99-35456af1d991)

![image](https://github.com/user-attachments/assets/5c8af066-1d58-4e7e-b944-088f440f2b30)

![image](https://github.com/user-attachments/assets/61950c67-1fd9-46a9-a924-b8d207fd5940)

![image](https://github.com/user-attachments/assets/818fe5ff-f368-4a05-b051-228adbb7821a)

![image](https://github.com/user-attachments/assets/2a045f3d-972b-4f97-839e-eb2e1acbd865)

![image](https://github.com/user-attachments/assets/b23f109d-710c-455d-9c4e-567d66f357ab)

![image](https://github.com/user-attachments/assets/db3d9dbe-d4f6-4684-866b-0a2d3b61c521)

![image](https://github.com/user-attachments/assets/002bf8bd-799c-4803-acb5-3e7fea37e2bd)

![image](https://github.com/user-attachments/assets/9e41e415-6436-4c0a-8676-81ba495c50a7)

![image](https://github.com/user-attachments/assets/e3f82bd2-9ee7-4c90-b78b-08265b5a289e)

![image](https://github.com/user-attachments/assets/38e214d7-1188-4e29-971a-e64a69ba5f01)

![image](https://github.com/user-attachments/assets/19b58b3b-f45a-4753-afe3-d96bcff0998b)

![image](https://github.com/user-attachments/assets/3358ad1d-0dd7-4240-992e-d29ba52f19be)

![image](https://github.com/user-attachments/assets/b8d87f07-f837-40d7-b9a8-4490e9429b2e)

![image](https://github.com/user-attachments/assets/03551ce3-001c-473e-8b34-fd7ef038cf42)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
