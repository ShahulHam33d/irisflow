import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("iris_data.csv")

#df.head()

#df.describe()

sdf = df.values
df.isnull().sum()

x = df.iloc[:, :4].values
y = df.iloc[:, 4:].values

from sklearn.preprocessing import LabelEncoder

labelen = LabelEncoder()
y = labelen.fit_transform(y)

'''train_x = x[:120,:]
train_y = y[:120]
test_x = x[120:,:]
test_y = y[120:].reshape(-1,1)'''

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(train_x, train_y)
pickle.dump(regressor,open("model.pkl","wb"))

model = pickle.load(open("model.pkl","rb"))

#print(model.predict([2]))
#pred_y = model.predict(test_x)
#pred_y = pred_y.reshape(-1, 1)


