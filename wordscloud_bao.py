import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import chardet
import jieba

from PIL import Image
from collections import Counter
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import cm

df = pd.read_csv(r"D:\data\聊天记录\2\utf8.csv", sep=',')

# Create a word cloud object with a specified Chinese font path
font_path = 'C:/Windows/Fonts/STXIHEI.TTF'  

df['Date'] = pd.to_datetime(df['StrTime'])  
df_filtered = df[(df['Type'] == 1) & (df['IsSender'] == 0) & (df['Date'] >= '2023-03-15')]

# Create a word cloud object with a specified Chinese font path
font_path = 'C:/Windows/Fonts/STXIHEI.TTF'  

# Save filtered content to a txt file
output_file_path = 'wordCloud_content/bao_content.txt'
df_filtered['StrContent'].to_csv(output_file_path, index=False, header=False, sep='\t')

with open('wordCloud_content/bao_content.txt', 'r', encoding='utf-8') as file:
    bao_text = file.read()

# Load stop words
stopwords_path = "wordCloud_content/stopwords.txt"

with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip() for line in f.readlines()]

# Segmentation of filtered content
bao_words = jieba.cut(bao_text)

# Define wordCloud shapes
mask_image_path = "wordCloud_content/blue.jpg" 
mask_image = np.array(Image.open(mask_image_path))
img_colors = ImageColorGenerator(mask_image) # type: ignore

# Define word cloud parameters
wordcloud = WordCloud(
    font_path=font_path,
    min_font_size = 6,
    max_font_size = 90,
    margin = 2,
    scale = 2,
    random_state = 42,
    width=800,
    height=800,
    background_color='white',
    stopwords=stopwords_list,
    mask=mask_image,  
).generate(' '.join([word for word in bao_words if word not in stopwords_list])) # type: ignore

wordcloud.recolor(color_func=img_colors)

output_image_path = "figures/bao_wordcloud.png"
wordcloud.to_file(output_image_path)

plt.figure(figsize=(40, 40))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()