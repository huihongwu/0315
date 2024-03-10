import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import calendar

# Read the dataset
df2 = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',')
df2['month'] = pd.to_datetime(df2['StrTime']).dt.month
df2 = df2[pd.to_datetime(df2['StrTime']) >= '2023-03-15']
month_counts = df2['month'].value_counts().sort_index()
scaled_sizes = month_counts * 0.15

df2['month_hui'] = pd.to_datetime(df2[df2['IsSender'] == 1]['StrTime']).dt.month
df2['month_bao'] = pd.to_datetime(df2[df2['IsSender'] == 0]['StrTime']).dt.month

labels = ['bao', 'hui']
colors = ['#C6DCE4','#8c85be']  

month_counts_hui = df2['month_hui'].value_counts().sort_index()
month_counts_bao = df2['month_bao'].value_counts().sort_index()

ax = month_counts_bao.plot(kind='bar', color=colors[0], label='bao', edgecolor='None')  
month_counts_hui.plot(kind='bar', color=colors[1], label='hui', edgecolor='None')

plt.xlabel('Month', fontname='Georgia', fontsize=15)
plt.ylabel('Messages', fontname='Georgia', fontsize=15)
plt.xticks(range(0, 12), [calendar.month_abbr[i] for i in range(1, 13)], fontname='Georgia', fontsize=15)
plt.yticks(fontname='Georgia', fontsize=15)

plt.grid(True, linestyle='solid', linewidth=0.5,  color='None')

font_prop = FontProperties(family='Georgia')
plt.legend(labels, loc="upper right", prop=font_prop)  

for i, v in enumerate(month_counts_bao):
    ax.text(i, v + 1, str(v), color='black', ha='center', va='bottom', fontsize=10, fontname='Georgia')

for i, v in enumerate(month_counts_hui):
    ax.text(i, v + 1, str(v), color='black', ha='center', va='bottom', fontsize=10, fontname='Georgia')

plt.tight_layout()

fig = plt.gcf()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
fig.set_size_inches(12, 7)
fig.savefig('figures/chat_bar_plot.png', dpi=100)
plt.show()
