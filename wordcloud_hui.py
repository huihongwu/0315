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

# Filter StrContent that meets Type == 1 and IsSender == 1
hui_content = df[(df['Type'] == 1) & (df['IsSender'] == 1)]['StrContent']
output_file_path = 'wordCloud_content/hui_content.txt'

# Save filtered content to a txt file
hui_content.to_csv(output_file_path, index=False, header=False, sep='\t')

with open('wordCloud_content/hui_content.txt', 'r', encoding='utf-8') as file:
    hui_text = file.read()

# Load stop words
stopwords_path = "wordCloud_content/stopwords.txt"

with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip() for line in f.readlines()]

# Segmentation of filtered content
hui_words = jieba.cut(hui_text)

# Define wordCloud shapes
mask_image_path = "wordCloud_content/heart.jpg" 
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
).generate(' '.join([word for word in hui_words if word not in stopwords_list])) # type: ignore

wordcloud.recolor(color_func=img_colors)

output_image_path = "figures/hui_wordcloud.png"
wordcloud.to_file(output_image_path)

plt.figure(figsize=(40, 40))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()