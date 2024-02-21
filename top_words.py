import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import jieba
from matplotlib.font_manager import FontProperties
from collections import Counter

font_path = 'C:/Windows/Fonts/simsun.ttc'  # Change to the path of a suitable font file

# Set the default font for the entire plotting environment
matplotlib.rcParams['font.family'] = 'SimSun'  

df = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',', encoding='utf-8')  # Specify UTF-8 encoding

df['Date'] = pd.to_datetime(df['StrTime'])  
df_filtered = df[(df['Type'] == 1) & (df['IsSender'] == 0) & (df['Date'] >= '2023-03-15')]

# Save filtered content to a txt file
output_file_path = 'wordCloud_content/bao_content.txt'
df_filtered['StrContent'].to_csv(output_file_path, index=False, header=False, sep='\t', encoding='utf-8')  # Specify UTF-8 encoding

with open('wordCloud_content/bao_content.txt', 'r', encoding='utf-8') as file:
    bao_text = file.read()

# Load stop words
stopwords_path = "wordCloud_content/stopwords.txt"

with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip() for line in f.readlines()]

# Segmentation of filtered content
bao_words = jieba.cut(bao_text)

# Count word occurrences
bao_word_count = Counter([word for word in bao_words if word not in stopwords_list])

# Select the top 30 words
bao_top_words = bao_word_count.most_common(30)

# Remove '\n' and ' ' from the top words
bao_top_words = [word for word in bao_top_words if word[0] not in {'\n', ' '}]

# Create a DataFrame for the top words
top_words_bao = pd.DataFrame(bao_top_words, columns=['Word', 'Count'])

#########################################################################################################

df_filtered = df[(df['Type'] == 1) & (df['IsSender'] == 1) & (df['Date'] >= '2023-03-15')]

# Save filtered content to a txt file
output_file_path = 'wordCloud_content/hui_content.txt'
df_filtered['StrContent'].to_csv(output_file_path, index=False, header=False, sep='\t', encoding='utf-8')  # Specify UTF-8 encoding

with open('wordCloud_content/hui_content.txt', 'r', encoding='utf-8') as file:
    hui_text = file.read()

stopwords_path = "wordCloud_content/stopwords.txt"
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip() for line in f.readlines()]

hui_words = jieba.cut(hui_text)
hui_word_count = Counter([word for word in hui_words if word not in stopwords_list])
hui_top_words = hui_word_count.most_common(30)
hui_top_words = [word for word in hui_top_words if word[0] not in {'\n', ' '}]
top_words_hui = pd.DataFrame(hui_top_words, columns=['Word', 'Count'])


# Plot a horizontal bar chart
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
blues_reversed = sns.color_palette('PuBu', n_colors=30)[::-1]
sns.barplot(x='Count', y='Word', data=top_words_bao, palette=blues_reversed)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().invert_xaxis() 
plt.xticks([])
plt.yticks([])  
plt.xlabel('') 
plt.ylabel('')
for i, (word, count) in enumerate(top_words_bao.iterrows()):
    plt.text(count['Count'], i, f"{count['Word']} ({count['Count']})", va='center', ha='right')

plt.subplot(1, 2, 2)
RdPu_reversed = sns.color_palette('BuPu', n_colors=30)[::-1]
sns.barplot(x='Count', y='Word', data=top_words_hui, palette=RdPu_reversed)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([])
plt.yticks([])  
plt.xlabel('') 
plt.ylabel('')
for i, (word, count) in enumerate(top_words_hui.iterrows()):
    plt.text(count['Count'], i, f"{count['Word']} ({count['Count']})", va='center', ha='left')
plt.tight_layout()
plt.savefig('figures/top30words.png',dpi=100)
plt.show()