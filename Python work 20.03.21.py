import numpy as np 
import pandas as pd

# read data from csv to dataframe 
df=pd.read_csv('mushrooms.csv')
print (df.head())
print (df.tail())
df1 = df #create a df1 for data cleaning copy dataframe 

#describe the dataset 
print(df.describe())

print(df.index)
print(df.index.max())
print(df.index.size)
print(df.columns)

print(df.index)
print(df.index.max())
print(df.index.size)
print(df.columns)

#data cleaning
#there are some data missing in column stalk-root, which represents in "?", it should be deleted? But there are 2480 results have the missing type of stalk-root
df[df['stalk-root'] == "?"]

#this is the dataset without all missing value in column stalk-root
df1[~ (df1['stalk-root'] == "?")]

#describe the dataset 
df1 = df1[~ (df1['stalk-root'] == "?")]
print(df1.describe())

df1.tail()
#from the result of df1.tail() we can see after cleaned the missing data, the index should be regenerated

#change the index 
df1.reset_index(drop=True, inplace=True)
print(df1.tail())

#after cleaned the missing data, we need to re-indexing the data. Put the order randomly before we seperate the dataset into two parts for machine learning traning.
random = np.random.permutation(df1.index.size)
print(random)
df2 = df1.take(random)

#also it can be good to change the initital letters back to the full word, as it need go back to the reference table to check what initials means is really annoying,

#separate the data to two part for machine learning 
#using Kfold split might be a better option than just random split the dataset into two parts? idea from: https://blog.csdn.net/MDbabyface/article/details/83271612 (Chinese only, but you can goole what is Kfold sorry)
from sklearn.model_selection import train_test_split
df2_train, df2_test = train_test_split(df2, test_size=0.3) #put ramdon_state=0 into the bracket, it also can separate the dataset randomly, but we might keep the code above to waste some word counts

df2_test.to_csv('test.csv')
df2_train.to_csv('test.csv')