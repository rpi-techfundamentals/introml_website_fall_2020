[![AnalyticsDojo](https://github.com/rpi-techfundamentals/fall2018-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Fastai - Revisiting Titanic</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>


from fastai import *
from fastai.tabular import * 
import numpy as np
import pandas as pd
import pandas as pd

train= pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/test.csv')



#Create a categorical variable from the family count 
def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'


for df in [train, test]:
    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]
    df['Title'] = df['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    df['Embarked']=df['Embarked'].fillna('S')
    df['NameLength'] = df['Name'].map(lambda x: len(x))
    df['FamilyS'] = df['SibSp'] + df['Parch'] + 1
    df['FamilyS'] = df['FamilyS'].apply(family)
train.isnull().sum(axis = 0)


train.head()

dep_var = 'Survived'
cat_names = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilyS']
cont_names = ['Age', 'Fare', 'SibSp', 'Parch', 'NameLength']
procs = [FillMissing, Categorify, Normalize]
test_data = (TabularList.from_df(test, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs))




data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           .label_from_df(cols=dep_var)
                           .add_test(test_data, label=0)
                           .databunch())

#Shows the Data
data.show_batch()



#Define our Learner
learn = tabular_learner(data, layers=[300,100], metrics=accuracy)


learn.lr_find()
learn.recorder.plot()

#fit the learner
learn.fit(7, 1e-2)  #Number of epocs and the learning rate. learn.save('final_train')

#Show the results
learn.show_results()

#This will get predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = to_np(np.argmax(predictions, 1))
labels.shape




#Writing to File
submission=pd.DataFrame(test.loc[:,['PassengerId']])
submission['Survived']=labels
#Any files you save will be available in the output tab below

submission.to_csv('submission.csv', index=False)

from google.colab import files
files.download('submission.csv')