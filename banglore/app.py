import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Bengaluru_House_Data.csv')
dataset[dataset==np.inf]=np.nan
dataset.fillna(dataset.mean(), inplace=True)
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
xc=X
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


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    a=float(int_features[-1])
    f=int_features

    int_features1 = int_features
    int_features[0] = x_label.transform(np.array([int_features[0]]))
    int_features = onehotencoder.transform([int_features]).toarray()
    #int_features = int_features[1:]
    #for i in int_features:
     #  int_features1.append(float(i)) 
    final_features = int_features
    if(f[1]<f[3] and f[1]<f[4]):
            return render_template('index.html', prediction_text='Error: ',prediction_text1='Check the value you used for no. of bathrooms and no. of Balcony',prediction_text2='')

        
        
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)    
    f1=int_features1[1]
 
    plt.scatter(xc[:,1],y,facecolors='none',edgecolors='blue')
    plt.scatter(f1,output,color='red')
    plt.title('no. of rooms vs Price')
    plt.xlabel('rooms')
    plt.ylabel('Price (in lakhs)')
    plt.savefig("static/pic11.jpg")
    plt.clf()

    f1=int_features1[2]
    j=[]
    for i in xc[:,2]:
        j.append(float(i))
    plt.scatter(j,y,facecolors='none',edgecolors='blue')
    plt.scatter(f1,output,color='red')
    plt.title('Area vs Price')
    plt.xlabel('Area (in Sq. ft.)')
    plt.ylabel('Price (in Lakhs)')
    plt.savefig("static/pic22.jpg")
    plt.clf()    
    
    f1=int_features1[3]
 
    plt.scatter(xc[:,3],y,facecolors='none',edgecolors='blue')
    plt.scatter(f1,output,color='red')
    plt.title('no. of bathrooms vs Price')
    plt.xlabel('bathrooms')
    plt.ylabel('Price (in Lakhs)')
    plt.savefig("static/pic33.jpg")
    plt.clf()

    f1=int_features1[4]
 
    plt.scatter(xc[:,4],y,facecolors='none',edgecolors='blue')
    plt.scatter(f1,output,color='red')
    plt.title('no. of balcony vs Price')
    plt.xlabel('balcony')
    plt.ylabel('Price (in Lakhs)')
    plt.savefig("static/pic44.jpg")
    plt.clf()

    f1=int_features1[5]
 
    plt.scatter(xc[:,5],y,facecolors='none',edgecolors='blue')
    plt.scatter(f1,output,color='red')
    plt.title('Age vs Price')
    plt.xlabel('Age (Years)')
    plt.ylabel('Price (in Lakhs)')
    plt.savefig("static/pic44.jpg")
    plt.clf()
    output = round(prediction[0]*100000, 2)

    return render_template('index.html', prediction_text='Present House Price Should Be Rs {}'.format(output),prediction_text1='Initial House Price Should Be Rs {}'.format(output-(output*0.013*a)),prediction_text2='Future House Price Should Be Rs {}'.format(output+(output*0.013*a)))

@app.route('/visual')
def visual():
    return render_template('visualize.html')

"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)"""

if __name__ == "__main__":
    app.run(debug=True)