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

df2 = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',', usecols=[4,7,8])
df2['Date'] = pd.to_datetime(df2['StrTime']).dt.date


# Filter data from March 15, 2023, onwards
start_date = pd.to_datetime('2023-03-15').date()
df2_filtered = df2[df2['Date'] >= start_date]

# Statistics on the number of chats per day
daily_counts = df2_filtered['Date'].value_counts().reset_index()
daily_counts.columns = ['Date', 'Chat_Count']

# To make a heat map, we may need to reshape the dates to create a matrix
# Create a matrix using pivot_table
heatmap_data = daily_counts.pivot_table(index='Date', values='Chat_Count', aggfunc='sum')

# Heat mapping using seaborn
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap="Blues",xticklabels=False)

plt.title('Chat Counts Heatmap', fontname='Georgia',fontsize=22)
plt.ylabel('Date', fontname='Georgia',fontsize=20)
plt.yticks(fontname='Georgia')

plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/heatmap_1.png',dpi=100)
plt.show()

df2['Date'] = pd.to_datetime(df2['StrTime'])
df2['Month'] = df2['Date'].dt.month  

heatmap_data = df2.pivot_table(index=df2['Date'].dt.day, columns='Month', values='StrTime', aggfunc='count') # type: ignore
sns.heatmap(heatmap_data, cmap="PuBu", linewidths=0.5, linecolor='gray')

plt.title('Monthly Chat Counts Heatmap', fontname='Georgia',fontsize=22)
plt.xlabel('Month', fontname='Georgia',fontsize=20)
plt.ylabel('Day of Month', fontname='Georgia',fontsize=20)
plt.xticks(fontname='Georgia',fontsize=15)  #Setting the x-axis labels
plt.yticks(fontname='Georgia',fontsize=15)
plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(15,8)
fig.savefig('figures/heatmap_2.png',dpi=100)
plt.show()
