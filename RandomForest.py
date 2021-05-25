import numpy as np
import pandas as pd  
  
#importing datasets  
data_train = pd.read_csv('CSV_DUMP_1.csv')
data_test = pd.read_csv('CSV_DUMP_2.csv')
x_train = data_train.iloc[:, [1,2,3,4,5]].values
y_train = data_train.iloc[:, 0].values 

x_test = data_test.iloc[:, [1,2,3,4,5]].values
y_test = data_test.iloc[:, 0].values  
  
# Splitting the dataset into training and test set.  
# from sklearn.model_selection import train_test_split  
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  
# print(np.count_nonzero(y_train))
# print(np.count_nonzero(y_test))

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)    

# Training
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  

#Predicting
y_pred= classifier.predict(x_test)  
y_check = np.vstack((y_pred,y_test))
y_check=y_check.transpose()
np.savetxt("out_RF.csv", y_check,fmt='%10.0f')

#Confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)