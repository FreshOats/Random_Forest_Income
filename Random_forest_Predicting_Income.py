## Predicting income using Random Forests
# Uses census information from UCI Machine Learning Repo
# Want to predict whether person makes more than $50k

## Headers of dataset
## age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#income_data = pd.read_csv('income.csv', header = 0)

# Take a look at the first row after the header. The "income category gives the number of <=50k"
#print(income_data.iloc[0])

## Notice after the first row, the rest have a space in front of each header because there is a space afer each comma. Need to fixt this delimiter in the initial read to include the comma and space as the delimiter
income_data = pd.read_csv('income.csv', header = 0, delimiter = ', ')

#print(income_data.iloc[0])

#Create a var that contains only the column 'income' from the income_data df
labels = income_data[['income']]
#data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex']]

#Remove sex as a label until fixed
#data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]

# Clean the Data!! change Male/Female by adding a new column and making it 0/1
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)
# This uses a lambda to create a new row called 'sex-int' by applying to the 'sex' row the if statement 0 if Male and 1 otherwise to each row. 

data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int']]


# Create training and test sets for the supervised learning
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

## Plant the random forest!
forest = RandomForestClassifier(random_state = 1) # Creates the classifier without a specified number of trees

forest.fit(train_data, train_labels) # This throws up an error code because 'sex' is labeled with strings Male and Female and needs numerical values.

score = forest.score(test_data, test_labels)
print(score) ## Score is only 82.5% right now, going to add native country

#print(income_data['native-country'].value_counts())  ## USA has 29000 to the next highest with only 643 - should probably split USA from everything else

# Use another lambda to establish USA 0 and other country 1
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United States' else 1)

# Add country-int to the data
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

## Plant the new forest!
forest = RandomForestClassifier(random_state = 1) 
forest.fit(train_data, train_labels) 
score = forest.score(test_data, test_labels)
print(score) ## Score is now 82.4% right - didn't add much value