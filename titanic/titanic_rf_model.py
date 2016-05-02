import numpy as np
import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier

def convertAge(df):
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == j+1)]['Age'].dropna().median()

    for i in range(0,2):
        for j in range(0,3):
            df.loc[(df.Age.isnull()) & (df.Gender==i) & (df.Pclass==j+1), 'AgeFill'] = median_ages[i][j]
    return df

def convertEmbarked(df):
    # missing value, fill na with most often occured value
    if (len(df[df.Embarked.isnull()]) > 0):
        df.loc[df.Embarked.isnull(), 'Embarked'] = df.Embarked.dropna().mode().iloc[0]
    ports = list(enumerate(np.unique(df.Embarked)))
    port_dict = { name: i for i, name in ports }
    df.Embarked = df.Embarked.map( lambda x: port_dict[x]).astype(int)
    return df

def preprocessData(df):
    # female = 0, male = 1, Sex -> Gender
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # SibSp + Parch = FamilySize
    df['FamilySize'] = df.SibSp + df.Parch
    # Age
    df['AgeFill'] = df.Age
    df = convertAge(df) 
    # Embarked
    df = convertEmbarked(df)  

    # Fare
    if len(df.Fare[df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):
            median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
        for f in range(0,3):
            df.loc[(df.Pclass == f+1) & (df.Fare.isnull()), 'Fare'] = median_fare[f]
    
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Age'], axis=1)
    return df
    
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# save PassengerId
ids = test_df['PassengerId'].values
# preprocess data
train_df = preprocessData(train_df)
test_df = preprocessData(test_df)
train_data = train_df.values
test_data = test_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("result.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'