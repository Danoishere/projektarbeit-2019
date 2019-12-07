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


path = 'singleworker_manyworker_comparison/'

t1 = pd.read_csv(path + 'analysis_round_140_manyworker.csv', delimiter=',')
t2 = pd.read_csv(path + 'analysis_round_219_singleworker.csv', delimiter=',')

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

group_by_round(t1, '7 workers')
group_by_round(t2, '1 worker')

plt.ylabel('Number of agents arriving')
plt.xlabel('Number of evaluation rounds (20 episodes/round)')
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig(path + 'singlemany_comparison.pgf')








