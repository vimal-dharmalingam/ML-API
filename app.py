#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import re
app = Flask(__name__)
model = pickle.load(open('ml_model//Rf_model_version1.pkl','rb'))
Raw_data=pd.read_csv('test.csv')

@app.route('/api',methods=['POST','GET'])
def predict():
    data=Raw_data.drop(['PassengerId'], axis=1)
    data['relatives'] = data['SibSp'] + data['Parch']
    data.loc[data['relatives'] > 0, 'not_alone'] = 0
    data.loc[data['relatives'] == 0, 'not_alone'] = 1
    data['not_alone'] = data['not_alone'].astype(int)
    # Converting Cabin feature in to Deck and drop cabin feature
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data['Cabin'] = data['Cabin'].fillna("U0")
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(deck).fillna(0).astype(int) 
    data = data.drop(['Cabin'], axis=1)
    mean = 3.46
    std = data["Age"].replace(np.NaN,0).std()
    is_null = data["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = data["Age"].astype(int)
    data
    #
    ## Embarked
    common_value = 'S'
    data['Embarked'] = data['Embarked'].fillna(common_value)
    data['Fare'] = data['Fare'].fillna(0).astype(int)
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    # extract titles
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    data['Title'] = data['Title'].map(titles)
    # filling NaN with 0, to get safe
    data['Title'] = data['Title'].fillna(0)
    data = data.drop(['Name'], axis=1)

    genders = {"male": 0, "female": 1}
    data['Sex'] = data['Sex'].map(genders)

    data = data.drop(['Ticket'], axis=1)
    data


    ports = {"S": 0, "C": 1, "Q": 2}
    data['Embarked'] = data['Embarked'].map(ports)

    ## Convert numeric into categorical features
    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 4
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 5
    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6
    data.loc[ data['Age'] > 66, 'Age'] = 6

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare']   = 3
    data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare']   = 4
    data.loc[ data['Fare'] > 250, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)

    data['Age_Class']= data['Age']* data['Pclass']
    data['Fare_Per_Person'] = data['Fare']/(data['relatives']+1)
    data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)

    Saved_model=pickle.load(open('Rf_model_version1.pkl','rb'))
    Saved_model

    Prediction_output=Saved_model.predict(data)

    Results=pd.DataFrame()
    Results['Predicted_result']=Prediction_output
    Results=Results.set_index(Raw_data['PassengerId'])
    Results.to_csv('MOdel_results_28082020.xlsx')
    
    cust_ids=Raw_data['PassengerId']
    # result=[Prediction_output]
    #jsonized = map(lambda item: {'Cust_ids':item[0], 'Prediction':item[1]}, zip(cust_ids,result))
    
    output = {'Prediction_Output': [{'Cust_id': a, 'Prediction': t} for a, t in zip(list(np.array(cust_ids).tolist()),Prediction_output.tolist())]}
    print(output)
    
    return output

if __name__ == '__main__':
    app.run(port=5000, debug=True,use_reloader=False)

