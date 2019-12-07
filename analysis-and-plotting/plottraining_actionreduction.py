import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

t1 = pd.read_csv('train_withactionreduction.csv', delimiter=',')
t2 = pd.read_csv('train_noactionreduction.csv', delimiter=',')

def group_by_round(df:pd.DataFrame, label):

    x = df['Step'].to_numpy()
    y = df['Value'].to_numpy()

    num = np.count_nonzero(x < 10000)

    x = x[:num]
    y = y[:num]
    y *= 14

    plt.plot(x, y, label=label)

group_by_round(t1, 'With action reduction')
group_by_round(t2, 'Without action reduction')

plt.ylabel('Number of agents arriving')
plt.xlabel('Number training episodes')
plt.legend()
plt.show()






