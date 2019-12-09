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


path = 'action_reduction_comparison/'
plt.rcParams["figure.figsize"] = (5,3)

t1 = pd.read_csv(path + 'train_withactionreduction.csv', delimiter=',')
t2 = pd.read_csv(path + 'train_noactionreduction.csv', delimiter=',')

t1['time'] = pd.to_datetime(t1['Wall time'],unit='s')
t2['time'] = pd.to_datetime(t2['Wall time'],unit='s')

t1_start = t1['time'].min()
t2_start = t2['time'].min()

def group_by_round(df:pd.DataFrame, label, time_start):
    df['elapsed_hrs'] = df['time'] - time_start
    df = df[df['elapsed_hrs'] < pd.Timedelta('0 days 12 hours')]
    df['elapsed_hrs'] = pd.to_datetime(df['elapsed_hrs'])

    x = df['elapsed_hrs'].to_numpy()
    y = df['Value'].to_numpy()

    plt.plot(x, y, label=label)

group_by_round(t1, 'With action reduction', t1_start)
group_by_round(t2, 'Without action reduction', t2_start)

plt.ylabel('Percentage of agents arriving')
plt.xlabel('Hours of training')

myFmt = matplotlib.dates.DateFormatter("%Hh")
plt.gca().xaxis.set_major_formatter(myFmt)
plt.gcf().autofmt_xdate()

plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig(path + 'comparison_action_reduction.pgf')




