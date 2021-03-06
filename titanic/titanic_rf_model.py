import numpy as np
import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import string

def convertAge(df):
    le = preprocessing.LabelEncoder()
    median_ages = np.zeros((2,3))
    mean_ages = np.zeros(4)
    mean_ages[0] = df[df.Title=='Miss']['Age'].dropna().mean()
    mean_ages[1] = df[df.Title=='Mrs']['Age'].dropna().mean()
    mean_ages[2] = df[df.Title=='Mr']['Age'].dropna().mean()
    mean_ages[3] = df[df.Title=='Mrs']['Age'].dropna().mean()

    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    le.fit(df['AgeCat'])
    cat_age = le.transform(df['AgeCat'])
    df['AgeCat'] = cat_age.astype(np.float)
    return df

def convertEmbarked(df):
    # missing value, fill na with most often occured value
    if (len(df[df.Embarked.isnull()]) > 0):
        df.loc[df.Embarked.isnull(), 'Embarked'] = df.Embarked.dropna().mode().iloc[0]
    ports = list(enumerate(np.unique(df.Embarked)))
    port_dict = { name: i for i, name in ports }
    df.Embarked = df.Embarked.map( lambda x: port_dict[x]).astype(int)
    return df

###utility to clean and munge data
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    #print big_string
    return np.nan

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title =='':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title
def preprocessData(df):
    #creating a title column from name
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df.apply(replace_titles, axis=1)

    # female = 0, male = 1, Sex -> Gender
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # SibSp + Parch = FamilySize
    df['FamilySize'] = df.SibSp + df.Parch
    df['Family'] = df.SibSp*df.Parch
    # Age
    df['AgeFill'] = df.Age
    df = convertAge(df) 
    # Embarked
    df = convertEmbarked(df)  

    # Fare
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    if len(df.Fare[df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):
            median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().mean()
        for f in range(0,3):
            df.loc[(df.Pclass == f+1) & (df.Fare.isnull()), 'Fare'] = median_fare[f]
    
    # Fare per person
    df['FarePerPerson'] = df.Fare/(df.FamilySize+1)
    df['AgeClass'] = df['AgeFill']*df['Pclass']
    df['ClassFare'] = df['Pclass']*df['FarePerPerson']
    le = preprocessing.LabelEncoder()
    le.fit(df.Title)
    title = le.transform(df.Title)
    df['Title'] = title.astype(np.float)
    le.fit(df.Sex)
    x_sex = le.transform(df.Sex)
    df['Sex'] = x_sex.astype(np.float)
    # Pclass Level
    # df['HighLow'] = df['Pclass']
    # df.loc[(df.FarePerPerson<8), 'HighLow'] = 'Low'
    # df.loc[(df.FarePerPerson>=8), 'HighLow'] = 'High'

    # le.fit(df.HighLow)
    # high_low = le.transform(df.HighLow)
    # df['HighLow'] = high_low.astype(np.float)
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

    le.fit( df['Ticket'])
    x_ticket=le.transform( df['Ticket'])
    df['Ticket']=x_ticket.astype(np.float)
    df = df.drop(['Name', 'Cabin', 'PassengerId', 'Age'], axis=1)
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