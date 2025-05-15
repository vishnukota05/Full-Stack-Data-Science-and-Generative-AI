# IMPORT LIBRARY
import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd	

# IMPORT THE DATASET

dataset = pd.read_csv(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\1. KODI WORK\1. NARESH\2. EVENING BATCH\N_Batch -- 7.00PM -- Ju25\3. MAR\18th - ML\5. Data preprocessing\Data.csv")

# INDEPENDENT VARIABLE
X = dataset.iloc[:, :-1].values	
# DEPENDENT VARIABLE
y = dataset.iloc[:,3].values  

# SKLEARN FILL MISSING NUMERICAL VALUE
from sklearn.impute import SimpleImputer

imputer = SimpleImputer() 

imputer = imputer.fit(X[:,1:3]) 

X[:, 1:3] = imputer.transform(X[:,1:3]) 

# IMPUTE CATEGORICAL VALUE FOR INDEPDENT 
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

## IMPUTE CATEGORICAL VALUE FOR DEPENDENT 

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

# SPLIT THE DATA 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,train_size=0.8, random_state=0) 


