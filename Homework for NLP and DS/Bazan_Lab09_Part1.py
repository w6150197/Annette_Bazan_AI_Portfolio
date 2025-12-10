#!/usr/bin/env python
# coding: utf-8

# # Lab 09 - Language Models- Annette Bazan (Part 1)

# ### Intoduction:
# 1. **What is a lg model?** A lg model is a technique that uses probabilities to predict the next sequence of a sentence (the next word).
# 
# 2. **Types of lg models: A) Statistical Lg Models:** These models use statistical techniques such as N-gram to predict and generate texts. There are two types of statistical lg models:**N-grams and exponential models.**
# 
# **B) Neural Lg Models:** This type is based on deep learning and neural networks models. This type performs very well in NLP. It can be at word or character level. There are two types of this models:**Recurrent Neural Networks(RNNs) and Transformers such as BERT and GPT**
# 
#  1. **Uses of Lg Models:**
#  - Text Generation
#  - Machine Translation
#  - Speech Recognition
#  - Sentiment Analysis
#  
# ### Building a Neural Network Model
#  - RNN (with sequential data)

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install jupyter tensorflow')
get_ipython().system('pip install ipykernel')


# In[3]:


# 1. importing the libraries

# general libraries
import numpy as np
import pandas as pd
import math

# data visualiztion libraries
import matplotlib.pyplot as plt
import seaborn as sns

# nltk libraries
import nltk
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
nltk.download('reuters')
nltk.download('punkt')


# tensorflow libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# deep learning libraries
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[ ]:


# 2. loading the data (text)
data_text = "The unanimous Declaration of the thirteen united States of\
America, When in the Course of human events, it becomes necessary for one\
people to dissolve the political bands which have connected them with another,\
and to assume among the powers of the earth, the separate and equal station to\
which the Laws of Nature and of Nature's God entitle them, a decent respect to\
the opinions of mankind requires that they should declare the causes which\
impel them to the separation.\
We hold these truths to be self-evident, that all men are created equal,\
that they are endowed by their Creator with certain unalienable Rights,\
that among these are Life, Liberty and the pursuit of Happiness.--That to\
secure these rights, Governments are instituted among Men, deriving their\
just powers from the consent of the governed, --That whenever any Form of\
Government becomes destructive of these ends, it is the Right of the People\
to alter or to abolish it, and to institute new Government,\
laying its foundation on such principles and organizing its powers in such\
form, as to them shall seem most likely to effect their Safety and\
Happiness. Prudence, indeed, will dictate that Governments long established\
should not be changed for light and transient causes; and accordingly all\
experience hath shewn, that mankind are more disposed to suffer, while evils\
are sufferable, than to right themselves by abolishing the forms to which they\
are accustomed. But when a long train of abuses and usurpations, pursuing\
invariably the same Object evinces a design to reduce them under absolute\
Despotism, it is their right, it is their duty, to throw off such Government,\
and to provide new Guards for their future security.--Such has been the patient\
sufferance of these Colonies; and such is now the necessity which constrains\
them to alter their former Systems of Government. The history of the present\
King of Great Britain is a history of repeated injuries and usurpations,\
all having in direct object the establishment of an absolute Tyranny over\
these States. To prove this, let Facts be submitted to a candid world.\
General Congress, Assembled, appealing to the Supreme Judge of the world\
for the rectitude of our intentions, do, in the Name, and by Authority of\
the good People of these Colonies, solemnly publish and declare, That these\
United Colonies are, and of Right ought to be Free and Independent States;\
that they are Absolved from all Allegiance to the British Crown, and that\
all political connection between them and the State of Great Britain,\
is and ought to be totally dissolved; and that as Free and Independent\
States, they have full Power to levy War, conclude Peace, contract\
Alliances, establish Commerce, and to do all other Acts and Things which\
Independent States may of right do. And for the support of this Declaration,\
with a firm reliance on the protection of divine Providence, we mutually\
pledge to each other our Lives, our Fortunes and our sacred Honor."


# In[ ]:


# 3. Data (text) Processing
import re # Regular Expressions: provides powerful tools to work with text patterns


# In[ ]:


# 4. defining a function for text_cleaner

def text_cleaner(text):
  newString = text.lower() # lower cast text
  newString = re.sub(r"'s\b", '', newString)
  newString = re.sub('[^a-zA-Z]', ' ', newString) # removing any puncatoins

  long_words = [] # removing short words and storing long words in this list

  for i in newString.split(): # loop in newString and split
    if len(i) >= 3: # any part that has more than 3 words
      long_words.append(i) # store it in long_words list

  return(' '.join(long_words).strip())


# In[ ]:


# 5. data preprocessing
# a) cleaning (processing) the text using the text_cleaner function
clean_text = text_cleaner(data_text)


# In[ ]:


# b) creating sequences
# let's define another function for creating sequences

def create_seq(text):
  length = 30 # the length of each sequence
  sequences = list()

  for i in range(length, len(text)):
    seq = text[i-length: i+1] # select sequence of tokens
    sequences.append(seq) # adding more sequences

  print('Total sequences: %d' % len(sequences))
  return sequences


# In[ ]:


# c. creating and displaying the number of sequences using the function create_seq
sequences = create_seq(clean_text)


# ### **Conclusion:**
# 1. Logistic regression performs well when the dataset is linearly separable.
# 
# 2. Decision trees can capture non-linear relationships but may overfit the training data.
# 
# 3. Random forests improve on decision trees by reducing overfitting through averaging multiple trees.
# 
# 4. Gradient boosting models tend to outperform random forests but are more computationally expensive.
# 
# 5. Support Vector Machines (SVMs) work best with a clear margin of separation between classes.
# 
# 6. Neural networks require more data and tuning but can model very complex patterns.
# 
# 7. Scaling input features is essential for models like logistic regression and SVMs.
# 
# 8. Overfitting can be mitigated using techniques like cross-validation and regularization.
# 
# 9. Confusion matrices and accuracy scores help evaluate classification performance.
# 
# 10. ROC curves and AUC scores offer deeper insight into model discrimination power.
# 
# 11. Feature importance metrics help interpret tree-based models like random forests and XGBoost.
# 
# 12. Hyperparameter tuning is crucial for maximizing model performance.
# 
# 13. Cross-validation ensures that models generalize well to unseen data.
# 
# 14. Ensemble methods generally perform better than single models on complex datasets.
# 
# 15. Choosing the right model depends on the data structure, size, and the problem at hand.

# ##### End of Lab 09 - Part 1
