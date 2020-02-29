#Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('Bengaluru_House_Data.csv')
dataset[dataset==np.inf]=np.nan
dataset.fillna(dataset.mean(), inplace=True)
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Converting Area into Square ft.
for i in range(len(X)):
    if("-" in str(X[i,2])):
        l=str(X[i,2]).split(" - ")
        X[i,2]=(float(l[0])+float(l[1]))/2     
    elif("Sq. Meter" in str(X[i,2])):
        l=str(X[i,2]).split("Sq. Meter")
        X[i,2]=float(l[0])*10.7639   
    elif("Perch" in str(X[i,2])):
        l=str(X[i,2]).split("Perch")
        X[i,2]=float(l[0])*272.25   
    elif("Sq. Yards" in str(X[i,2])):
        l=str(X[i,2]).split("Sq. Yards")
        X[i,2]=float(l[0])*9     
    elif("Acres" in str(X[i,2])):
        l=str(X[i,2]).split("Acres")
        X[i,2]=float(l[0])*43560     
    elif("Cents" in str(X[i,2])):
        l=str(X[i,2]).split("Cents")
        X[i,2]=float(l[0])*435.6    
    elif("Guntha" in str(X[i,2])):
        l=str(X[i,2]).split("Guntha")
        X[i,2]=float(l[0])*1089    
    elif("Grounds" in str(X[i,2])):
        l=str(X[i,2]).split("Grounds")
        X[i,2]=float(l[0])*2400
    else:
        X[i,2]=float(X[i,2])
   
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x_label = LabelEncoder()
X[:,0] = x_label.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray() 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_train)

# save the model to disk
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


