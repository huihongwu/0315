import re
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import nltk
from snownlp import SnowNLP
from dateutil import parser
from nltk.sentiment import SentimentIntensityAnalyzer

from matplotlib.font_manager import *
from matplotlib import cm
from datetime import datetime
from scipy.stats import boxcox

# Read the dataset
df2 = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',')
df2['month'] = pd.to_datetime(df2['StrTime']).dt.month
df2 = df2[pd.to_datetime(df2['StrTime']) >= '2023-03-15']
month_counts = df2['month'].value_counts().sort_index()
scaled_sizes = month_counts * 0.15
 

# Assuming ['StrContent'] contains the message content
keywords = ['love', '拥抱','抱抱','like you','亲亲','爱', '我喜欢', '喜欢你']

# Create a new column indicating if each message contains any of the keywords
df2['ContainsKeyword'] = df2['StrContent'].str.contains('|'.join(keywords), case=False)

# Calculate the count of messages containing the keywords for each sender
count_contains_keyword = df2.groupby('IsSender')['ContainsKeyword'].sum()

# Creating Bar Plot
labels = ['bao', 'hui']
colors = ['#C6DCE4','#F2D1D1']

plt.figure(figsize=(10, 6))

# Bar plot for the count of messages containing keywords
plt.bar(labels, count_contains_keyword, color=colors)

# Display the count of messages containing keywords on each bar
for i, count in enumerate(count_contains_keyword):
    plt.text(i, count + 0.1, f"{count}", ha='center', fontsize=12)

plt.xlabel('Sender', fontname='Georgia', fontsize=14)
plt.ylabel('Number of Messages', fontname='Georgia', fontsize=14)
font_prop = FontProperties(family='Georgia')
fig = plt.gcf()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
fig.savefig('figures/loveWord_distribution.png', dpi=100)  # Save the bar plot with corrected file extension
plt.show()

#####################################################################################################################################################

# Scatterplotting
plt.figure(facecolor='white')
plt.xlabel('Month', fontname='Georgia',fontsize=20)
plt.ylabel('Messages', fontname='Georgia',fontsize=20)
plt.xticks(range(1, 13), fontname='Georgia',fontsize=15)
plt.yticks(fontname='Georgia',fontsize=15)
plt.scatter(month_counts.index, month_counts.values, color='#9EB8D9', marker='o',s=scaled_sizes) # type: ignore
fig = plt.gcf()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
fig.set_size_inches(15,8)
fig.savefig('figures/chat_month.png',dpi=100)
plt.show()

#####################################################################################################################################################

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

month_counts_bao.plot(kind='line', marker='o',markersize=14, linewidth=5, label='bao',color='#C6DCE4')
month_counts_hui.plot(kind='line', marker='o',markersize=14, linewidth=5, label='hui',color='#F2D1D1')

# Add labelled maximum value and corresponding month to the highest point
plt.annotate(f'Max: {max_bao}', xy=(max_month_bao, max_bao), xytext=(max_month_bao + 0.5, max_bao + 10), # type: ignore
             arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=1, color='lightgrey'),
             fontsize=12,fontname='Georgia',color='dimgray')

plt.annotate(f'Max: {max_hui}', xy=(max_month_hui, max_hui), xytext=(max_month_hui + 0.5, max_hui + 10), # type: ignore
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
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.legend(labels, loc="lower right", prop=font_prop)
fig.set_size_inches(15,8)
fig.savefig('figures/chat_plot.png',dpi=100)
plt.show()

#####################################################################################################################################################
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
plt.axis('equal')  # Keep the pie chart round

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_pie',dpi=100)
plt.show()

#####################################################################################################################################################

dates = pd.to_datetime(df2['StrTime'])
weekdays = dates.dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = weekdays.value_counts().reindex(weekday_order)


colors = ['#FFCACC', '#7C93C3', '#E3DFFD', '#D0F5BE', '#FFDEB4', '#F7A4A4', '#FFEECC']
max_day = weekday_counts.idxmax()
explode = [0.1 if day == max_day else 0 for day in weekday_counts.index]
plt.figure(figsize=(8, 8))

plt.pie(weekday_counts, explode=explode, labels=weekday_counts.index, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,textprops={'fontsize': 18}) # type: ignore
font_prop = FontProperties(family='Georgia')
#plt.legend(labels=weekday_counts.index, loc="lower right",prop=font_prop)
plt.axis('equal') 

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/chat_pie_2',dpi=100)
plt.show()

#####################################################################################################################################################
df2['hour'] = pd.to_datetime(df2['StrTime']).dt.hour

plt.xlabel('Time', fontname='Georgia',fontsize=18)
plt.ylabel('Number of messages', fontname='Georgia',fontsize=18)

sns.set_style('white')

sns.histplot(df2['hour'],bins=23,kde=True, color='#9EB8D9')

plt.xticks(np.arange(0, 24, 1.0), fontname='Georgia',fontsize=15)
plt.yticks(fontname='Georgia',fontsize=15)

fig = plt.gcf()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
fig.set_size_inches(15,8)
fig.savefig('figures/chat_time.png',dpi=100)
plt.show()

#####################################################################################################################################################
# Convert 'Date' column to datetime type
df2['Date'] = pd.to_datetime(df2['StrTime'])

df2.set_index('Date', inplace=True)

# Create a dictionary with monthly statistics
monthly_counts = {}

# Processing of the DataFrame for each month
for month in range(3, 13):
    month_str = f'2023-{month:02d}'
    month_df = df2.loc[month_str]
    daily_count = month_df.resample('D').size()
    monthly_counts[month_str] = daily_count

# Additional code for January and February 2024
for month in range(1, 3):
    month_str = f'2024-{month:02d}'
    month_df = df2.loc[month_str]
    daily_count = month_df.resample('D').size()
    monthly_counts[month_str] = daily_count

plt.figure(figsize=(1, 1))

# Add titles and tags
labels = ['2023-03', '2023-04', '2023-05', '2023-06', '2023-07','2023-08', '2023-09', '2023-10', '2023-11', '2023-12','2024-01', '2024-02']
colors = ['#FFCACC', '#7C93C3', '#E3DFFD', '#D0F5BE', '#FFDEB4', '#F7A4A4', '#ADD8E6', '#F2D1D1', '#C87AA4', '#7BAAF6', '#FF7AAC', '#AABB80']

for idx, (month, count_data) in enumerate(monthly_counts.items()):
    plt.plot(count_data.index.day, count_data.values, linestyle='solid', linewidth=2, color=colors[idx], label=month)

    # Find the maximum value and the corresponding index
    max_value = count_data.max()
    max_day = count_data.idxmax().day  
    
    # Maximum values marked on the graph
    plt.annotate(f'Max: {max_value}', xy=(max_day, max_value), xytext=(max_day + 1.2, max_value + 1),
                 arrowprops=dict(facecolor='black', arrowstyle='->',linewidth=2, color='grey'),
                 fontsize=13, fontname='Georgia',color='dimgray')
    
plt.xlabel('Day', fontname='Georgia',fontsize=20)
plt.ylabel('Messages', fontname='Georgia',fontsize=20)
plt.xticks(range(1, 32),fontname='Georgia',fontsize=15)  
plt.yticks(fontname='Georgia',fontsize=15)
font_prop = FontProperties(family='Georgia')
plt.legend(labels, loc="best",prop=font_prop)

plt.tight_layout()

fig = plt.gcf()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
fig.set_size_inches(15,8)
fig.savefig('figures/chat_plot2.png',dpi=100)
plt.show()
