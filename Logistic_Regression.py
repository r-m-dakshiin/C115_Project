import pandas as pd
import plotly.express as px


df = pd.read_csv("data.csv")
score_list = df["Score"].tolist()
accepted_list = df["Accepted"].tolist()
fig = px.scatter(x=score_list, y = accepted_list)
fig.show()

import numpy as np

score_array = np.array(score_list)
accepted_array = np.array(accepted_list)

#Slope and intercept using pre-built function of Numpy
m,c  = np.polyfit(score_array, accepted_array, 1)

y = []

for x in score_array:
    y_value = m*x + c
    y.append(y_value)

#plotting the graph
fig = px.scatter(x=score_array, y = accepted_array)
fig.update_layout(shapes = [
    dict(
        type = 'line',
        y0 = min(y),
        y1 = max(y),
        x0 = min(score_array),
        x1 = max(score_array)

    )
])
fig.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(score_list, (len(score_list), 1))
Y = np.reshape(accepted_list, (len(accepted_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y.ravel(), color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))




#Using the line formula 
X_test = np.linspace(0, 100, 200)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

#do hit and trial by changing the vlaue of X_test here.
plt.axvline(x=X_test[23], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0,30)
plt.show()
print(X_test[23])