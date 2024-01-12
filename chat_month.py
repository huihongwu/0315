import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import matplotlib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from matplotlib.font_manager import *
from matplotlib import cm
from datetime import datetime
from scipy.stats import boxcox

# Read the dataset
df2 = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',', usecols=[4,7,8])
df2['month'] = pd.to_datetime(df2['StrTime']).dt.month
month_counts = df2['month'].value_counts().sort_index()
scaled_sizes = month_counts * 0.15


# Scatterplotting
plt.figure(facecolor='white')
plt.xlabel('Month', fontname='Georgia',fontsize=20)
plt.ylabel('Messages', fontname='Georgia',fontsize=20)
plt.xticks(range(1, 13), fontname='Georgia',fontsize=15)
plt.yticks(fontname='Georgia',fontsize=15)
plt.scatter(month_counts.index, month_counts.values, color='#9EB8D9', marker='o',s=scaled_sizes) # type: ignore

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_month.png',dpi=100)
plt.show()


df2['month_hui'] = pd.to_datetime(df2[df2['IsSender'] == 1]['StrTime']).dt.month
df2['month_bao'] = pd.to_datetime(df2[df2['IsSender'] == 0]['StrTime']).dt.month

labels = ['bao', 'hui']
colors = ['#C6DCE4','#F2D1D1']

month_counts_hui = df2['month_hui'].value_counts().sort_index()
month_counts_bao = df2['month_bao'].value_counts().sort_index()

# # Find the maximum value of hui and bao for each month and the corresponding months
max_hui = month_counts_hui.max()
max_month_hui = month_counts_hui.idxmax()

max_bao = month_counts_bao.max()
max_month_bao = month_counts_bao.idxmax()

month_counts_bao.plot(kind='line', marker='o',markersize=10, linewidth=3, label='bao',color='#C6DCE4')
month_counts_hui.plot(kind='line', marker='o',markersize=10, linewidth=3, label='hui',color='#F2D1D1')

# Add labelled maximum value and corresponding month to the highest point
plt.annotate(f'Max: {max_bao}', xy=(max_month_bao, max_bao), xytext=(max_month_bao + 0.5, max_bao + 10), # type: ignore
             arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=1, color='lightgrey'),
             fontsize=12,fontname='Georgia',color='dimgray')

plt.annotate(f'Max: {max_hui}', xy=(max_month_hui, max_hui), xytext=(max_month_hui + 0.4, max_hui + 10), # type: ignore
             arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=1, color='lightgrey'),
             fontsize=12,fontname='Georgia', color='dimgray')

plt.xlabel('Month', fontname='Georgia',fontsize=20)
plt.ylabel('Messages', fontname='Georgia',fontsize=20)
plt.xticks(range(1, 13), fontname='Georgia',fontsize=15)
plt.yticks(fontname='Georgia',fontsize=15)

plt.grid(True, linestyle='solid', linewidth=0.5, color='white')

font_prop = FontProperties(family='Georgia')
plt.legend(labels, loc="best",prop=font_prop,)

plt.tight_layout()  

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_plot.png',dpi=100)
plt.show()


value_counts = df2['IsSender'].value_counts()
percentages = 100. * value_counts / value_counts.sum()

# Creating Pie Charts
labels = ['bao', 'hui']
colors = ['#C6DCE4','#F2D1D1']
explode = (0.1, 0)  #Highlight the first slice

plt.figure(figsize=(8, 8))

# Define formatting functions for displaying data inside a pie chart
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return f"{pct:.1f}%\n({absolute:d})"

plt.pie(value_counts, explode=explode, labels=labels, colors=colors, 
        autopct=lambda pct: func(pct, value_counts), shadow=True, startangle=80, textprops={'style':'italic' , 'fontsize': 18})

font_prop = FontProperties(family='Georgia')
plt.legend(labels, loc="best",prop=font_prop)
plt.axis('equal')  # Keep the pie chart round

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_pie',dpi=100)
plt.show()


dates = pd.to_datetime(df2['StrTime'])
weekdays = dates.dt.day_name()
weekday_counts = weekdays.value_counts()


colors = ['#FFCACC', '#7C93C3', '#E3DFFD', '#D0F5BE', '#FFDEB4', '#F7A4A4', '#FFEECC']
explode = (0.1, 0, 0, 0, 0, 0, 0)  
plt.figure(figsize=(8, 8))

plt.pie(weekday_counts, explode=explode, labels=weekday_counts.index, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,textprops={'fontsize': 18}) # type: ignore
font_prop = FontProperties(family='Georgia')
plt.legend(labels=weekday_counts.index, loc="best",prop=font_prop)
plt.axis('equal') 

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_pie_2',dpi=100)
plt.show()


df2['hour'] = pd.to_datetime(df2['StrTime']).dt.hour

plt.xlabel('Time', fontname='Georgia',fontsize=18)
plt.ylabel('Number of messages', fontname='Georgia',fontsize=18)

sns.set_style('white')

sns.histplot(df2['hour'],bins=23,kde=True, color='#9EB8D9')

plt.xticks(np.arange(0, 24, 1.0), fontname='Georgia',fontsize=15)
plt.yticks(fontname='Georgia',fontsize=15)

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_time.png',dpi=100)
plt.show()

# Convert 'Date' column to datetime type
df2['Date'] = pd.to_datetime(df2['StrTime'])

df2.set_index('Date', inplace=True)

# Create a dictionary with monthly statistics
monthly_counts = {}

# Processing of the DataFrame for each month
for year in range(2023, 2024):
    for month in range(3, 13):
        month_str = f'{year}-{month:02d}'
        month_df = df2.loc[month_str]
        daily_count = month_df.resample('D').size()
        monthly_counts[month_str] = daily_count

plt.figure(figsize=(12, 8))

# Add titles and tags
labels = [f'{year}-{month:02d}' for year in range(2023, 2024) for month in range(3, 13)]
colors = ['#FFCACC', '#7C93C3', '#E3DFFD', '#D0F5BE', '#FFDEB4', '#F7A4A4', '#FF90C2', '#F2D1D1', '#DAEAF1', '#7C93C3', '#A25772', '#9EB8D9']

for idx, (month, count_data) in enumerate(monthly_counts.items()):
    plt.plot(count_data.index.day, count_data.values, marker='o', linestyle='-', color=colors[idx], label=month)
    
    # Find the maximum value and the corresponding index
    max_value = count_data.max()
    max_day = count_data.idxmax().day  
    
    # Maximum values marked on the graph
    plt.annotate(f'Max: {max_value}', xy=(max_day, max_value), xytext=(max_day + 1.2, max_value + 1),
                 arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=1, color='lightgrey'),
                 fontsize=12, fontname='Georgia',color='dimgray')

plt.xlabel('Day', fontname='Georgia',fontsize=20)
plt.ylabel('Messages', fontname='Georgia',fontsize=20)
plt.xticks(range(1, 32),fontname='Georgia',fontsize=15)  
plt.yticks(fontname='Georgia',fontsize=15)
font_prop = FontProperties(family='Georgia')
plt.legend(labels, loc="best",prop=font_prop)

plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_plot2.png',dpi=100)
plt.show()
"""