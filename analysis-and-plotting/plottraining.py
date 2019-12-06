import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

t1 = pd.read_csv('analysis_round_140_withlstm.csv', delimiter=',')
t2 = pd.read_csv('analysis_round_177_nolstm.csv', delimiter=',')
t3 = pd.read_csv('analysis_round_102_noactionreduction.csv', delimiter=',')

def group_by_round(df:pd.DataFrame, label):
    arrived = df['arrived'].to_numpy()
    arrived = np.maximum(arrived[1:] - arrived[:-1], 0)
    arrived = np.insert(arrived, 0, 0)
    df['arrived'] = arrived

    ndf = pd.DataFrame()
    ndf['round'] = df.groupby('round')['round'].max()
    ndf['arrived'] = df.groupby('round')['arrived'].mean()
    print(ndf)
    print()

    x = ndf['round'].to_numpy()
    y = ndf['arrived'].to_numpy()

    x = x[:140]
    y = y[:140]

    plt.plot(x, y, label=label)

group_by_round(t1, 'With LSTM')
group_by_round(t2, 'No LSTM')
group_by_round(t3, 'No action reduction')

plt.legend()
plt.show()






