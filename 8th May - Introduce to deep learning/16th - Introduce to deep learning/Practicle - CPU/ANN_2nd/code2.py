 
# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\kdata\Desktop\AMXWAM\AMXWAM DATASCIENCACADEMY\AMXWAM TRAINING PART\MACHINE LEARNING\ML DATASET\7.DEEP LEARNING\P14-Part8-Deep-Learning\Section 35 - Artificial Neural Networks (ANN)\Python\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #Sequenctial will initialize the neural network
from keras.layers import Dense #dense will build the hidden layer
#from keras.layers import LeakyReLU,PReLU,ELU #
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 11, init = 'he_uniform',activation='relu',input_dim = 11))
# 1st HL i will consider for 6 neurons, init - initiliazaton parameter, weights need to be initialise, 
#classifier.add(Dense(units = 10, kernel_initializer = 'he_normal',activation='relu',input_dim = 11))
#classifier.add(Dropout(0.3))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))
classifier.add(Dense(units = 11, kernel_initializer = 'he_normal',activation='relu'))
#classifier.add(Dropout(0.4))

# Adding the third hidden layer

classifier.add(Dense(units = 15, kernel_initializer = 'he_normal',activation='relu'))
#classifier.add(Dropout(0.2))

# Adding the output layer
#classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 10)
# reason why i take validation_split because test my data separately

# list all data in history

print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)

#**********************************************************************
