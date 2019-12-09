import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


plt.rcParams["figure.figsize"] = (5,3)

path = 'lstm_comparison/'

t1 = pd.read_csv(path + 'analysis_round_140_withlstm.csv', delimiter=',')
t2 = pd.read_csv(path + 'analysis_round_177_nolstm.csv', delimiter=',')

t1['time'] = pd.to_datetime(t1['time'])
t2['time'] = pd.to_datetime(t2['time'])

t1_start = t1['time'].min()
t2_start = t2['time'].min()

def group_by_round(df:pd.DataFrame, label, time_start):
    arrived = df['arrived'].to_numpy()
    arrived = np.maximum(arrived[1:] - arrived[:-1], 0)
    arrived = np.insert(arrived, 0, 0)
    df['arrived'] = arrived
    df['elapsed_hrs'] = df['time'] - time_start
    df = df[df['elapsed_hrs'] < pd.Timedelta('0 days 12 hours')]
    df['elapsed_hrs'] = pd.to_datetime(df['elapsed_hrs'])

    ndf = pd.DataFrame()
    ndf['round'] = df.groupby('round')['round'].max()
    ndf['arrived'] = df.groupby('round')['arrived'].mean()
    ndf['elapsed_hrs'] = df.groupby('round')['elapsed_hrs'].first()

    x = ndf['elapsed_hrs'].to_numpy()
    y = ndf['arrived'].to_numpy()

    plt.plot(x, y, label=label)

group_by_round(t1, 'With LSTM', t1_start)
group_by_round(t2, 'No LSTM', t2_start)

plt.ylabel('Number of agents arriving')
plt.xlabel('Hours of training')

myFmt = matplotlib.dates.DateFormatter("%H:%Mh")
plt.gca().xaxis.set_major_formatter(myFmt)
plt.gcf().autofmt_xdate()

plt.legend()
plt.tight_layout()
# plt.show()

plt.savefig(path + 'lstm_comparison.pgf')








