import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_reports = ['random_report.csv', 
    'model17_09_report.csv',
    'model20_27_report.csv',
    'model23_14_report.csv',
    'model00_11_report.csv',
    'model00_48_report.csv',
    'model07_32_report.csv',
    'model00_22_report.csv']

for report in plot_reports:
    df = pd.read_csv(report)
    df = df.groupby('evaluation_round')

    eval_groups = range(1,4)
    ratios = np.zeros(3)
    for i in range(0,3):
        ratios[i] = df.get_group(i+1)['agents_done'].sum()/df.get_group(i+1)['num_agents'].sum()

    plt.plot(eval_groups, ratios,label=report)

plt.legend()
plt.title('Durchschnittliche Abschlussrate für Agenten pro Evaluationsrunde')
plt.show()