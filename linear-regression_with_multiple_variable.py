# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

# Load The Data Frame
df = pd.read_csv('new.csv') 
df.experience = df.experience.fillna(0)

# Convert the word to equivalant number(Col=experience)
lst = []
for item in df.experience:
  try:
    lst.append(w2n.word_to_num(item))
  except:
    lst.append(item)

# fill the missing value of column test_score with their median
df.experience = lst
df.test_score = df.test_score.fillna(df.test_score.median())


# initialize the linear regression model and fit it
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']], df.salary)

# predict
reg.predict([[12, 10, 10]])
