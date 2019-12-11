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


path = 'final-evaluation/'

plt.rcParams["figure.figsize"] = (6,4)

df = pd.read_csv(path + 'analysis_1_to_30_agents.csv', delimiter=',')
departing = np.arange(1,31)

arriving = df.groupby('departed')['arrived']
q050 = arriving.median()
q025 = (q050 - arriving.quantile(0.25)).abs()
q075 = (q050 - arriving.quantile(0.75)).abs()

plt.errorbar(departing, q050, yerr=[q025, q075], fmt='o',capsize=3, color='black')
plt.bar(departing, q050)
plt.plot([1,30],[1,30], label='Max. possible arrivals', color='orange')

plt.ylabel('Number of agents arriving')
plt.xlabel('Number of agents departing')
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig(path + 'final_comparison_departing_arrival.pgf')






