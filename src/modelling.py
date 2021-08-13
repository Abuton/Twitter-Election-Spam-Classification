# -*- coding: utf-8 -*-
"""Modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Abuton/Twitter-Spam-Classification/blob/main/notebooks/Modelling.ipynb
"""

import chart_studio.plotly as py

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')



import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import xgboost as xgb
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

import nltk
from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score

# Defining the global variables for the color schemes we will incorporate
pblue = "#496595"
pb2 = "#85a1c1"
pb3 = "#3f4d63"
pg = "#c6ccd8"
pb = "#202022"
pbg = "#f4f0ea"

pgreen = px.colors.qualitative.Plotly[2]

# read the data
df = pd.read_csv('election_data.csv')

df.info()

df.isna().sum()

df.dropna(subset=['clean_text'], inplace=True)

# Finding maximum length of text message

print(np.max(df['clean_text'].apply(lambda x: len(x.split())).values))

"""### Exploratory Data Analysis"""

# Checking balance of dataset
grouped_df = df.groupby('category').count().values.flatten()

fig = go.Figure()

fig.add_trace(go.Bar(
        x=['not spam'],
        y=[grouped_df[0]],
        name='Safe',
        text=[grouped_df[0]],
        textposition='auto',
        marker_color=pblue
)
             )
fig.add_trace(go.Bar(
        x=['spam'],
        y=[grouped_df[1]],
        name='Spam',
        text=[grouped_df[1]],
        textposition='auto',
        marker_color=pg
))

fig.update_layout(
    title='Class distribution in the dataset')

fig.show()

# Creating series with length as index
# Sorting the series by index i.e. length
len_df_ham = df[df['category']=='not spam'].clean_text.apply(lambda x: len(x.split())).value_counts().sort_index()
len_df_spam = df[df['category']=='spam'].clean_text.apply(lambda x: len(x.split())).value_counts().sort_index()

# X-axis consists of the length of the msgs
# Y-axis consists of the frequency of those lengths

fig = go.Figure()
fig.add_trace(go.Scatter(
x=len_df_ham.index,
y=len_df_ham.values,
name='Safe',
fill='tozeroy',
marker_color=pblue))

fig.add_trace(go.Scatter(
x=len_df_spam.index,
y=len_df_spam.values,
name='Spam',
fill='tozeroy',
marker_color=pg
))

fig.update_layout(
    title='Frequency of Tweets lengths')
fig.update_xaxes(range=[0, 80])
fig.show()

def cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['clean_text'] = df['clean_text'].apply(cleaning)

nltk.download('stopwords')

# Removing stop words
stop_words = stopwords.words('english')
more = ['u', 'im', 'c']
stop_words = stop_words + more


def sw_rem(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

df['clean_text'] = df['clean_text'].apply(sw_rem)

"""Stemming - Omits the ends of words to achieve the goal correctly, this works most of the times and can also remove the derivational suffix

    Lemmatization - Working with a vocabulary and morphological analysis of words, removing inflectional endings only and returning the base and dictionary form of a word.

As we do not require much emphasis on words, we will focus more on stemming than lemmatization,.

Stemming algorithms

We have multiple algorithms to achieve our stemming goals, some of them are as follows:

    PorterStemmer - Fast and efficient. Strips off the end (suffix) to produce the stems. It does not follow linguistics but rather a set of 05 rules for diferent cases.

    SnowballStemmer - Generate a set of rules for any language. These are useful for non-english stemming tasks.

    LancasterStemmer - Iterative algorithm, uses about 120 rules, it tries to find an applicable rule by the last character of each word. The last character may be omitted or replaced.
"""

stems = nltk.SnowballStemmer('english')

def stemming(text):
    text = ' '.join(stems.stem(word) for word in text.split())
    return text

df['clean_text'] = df['clean_text'].apply(stemming)

# Creating a pipeline

def pipeline(text):
    text = cleaning(text)
    text = ' ' .join(word for word in text.split(' ') if word not in stop_words)
    text = ' '.join(stems.stem(word) for word in text.split(' '))
    return text

df['clean_text'] = df['clean_text'].apply(pipeline)

# Encoding the categorical target variable

le = LabelEncoder()
le.fit(df['category'])

df['label_num'] = le.transform(df['category'])
df.head()

"""
Vectorization

We currently have each text record in string format. We need to convert each of those records into a vector that our models can work with. We will first do this using the bag-of-words model.

We will use two major approaches here

    CountVectorizer - Working on frequency of each word in the given string.

    Term frequency-inverse document frqeuency TFIDF - Works on frequency divided by the appearance of the given word in the total documents.

"""

x = df['clean_text']
y = df['label_num']

len(x), len(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=201)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

"""## Applying Oversampling"""


over = SMOTE(sampling_strategy='auto')
under = RandomUnderSampler(sampling_strategy='auto')

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)

# First working with count vectorizer

# instantiate the vectorizer
count = CountVectorizer(stop_words='english', ngram_range=(1,2))
count.fit(x)

x_train_num = count.transform(x_train)
x_test_num = count.transform(x_test)

# transform the dataset
X_train_resample, y_train_resample = pipeline.fit_resample(x_train_num, y_train)

print(X_train_resample.shape)

"""

The CountVectorizer model can be tuned in a variety of ways:

    Stop words - Extremely common words can be omitted by the model by setting this parameter to the language corresponding to the text.

    ngram_range - It pairs up words together as features. If we consider bigrams and we have a sentence "I am happy", we will have two features - ["I am", "am happy"]. We can define a range of ngrams, so if we have the same sentence with a range from 1 to 2, our features will be: ["I", "am", "happy", "I am", "am happy"]. This increase is features helps to fine tune the model.

    min_df, max_df - Minimum and maximum frequencies of words of n-grams that can be used as features. If either of the conditions are not met, the feature will be omitted.

    max_features - Choose the most frequent words and drop everything else.

"""

# Example of a tuned model
count_tuned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=130)

# Working with TF-IDF now
from sklearn.feature_extraction.text import TfidfTransformer
# We are using transformer here
# If we use vectorizer, we can directly use the text
tfidf = TfidfTransformer()

tfidf.fit(X_train_resample)
x_train_tfidf = tfidf.transform(X_train_resample)

x_train_tfidf

# We will be creating seaborn and plotly confusion matrices

x_axes = ['Safe','Spam']
y_axes = ['Spam', 'Safe']

def conf_matrix(z, x=x_axes, y=y_axes):
    z = np.flip(z, 0)
    # Change each element of z to string 
    # This allows them to be used as annotations
    z_str = [[str(y) for y in x] for x in z]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_str)
    
    fig.update_layout(title_text='Confusion matrix', xaxis=dict(title='Predicted Value'),
                     yaxis=dict(title='Real value'))
    
    fig['data'][0]['showscale'] = True
    return fig


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
categories=['Safe', 'Spam']

def seaborn_conf(y, ypred):
    y_true = ["Not-Spam", "Spam"]
    y_pred = ["Not-Spam", "Spam"]

    cf = confusion_matrix(y, ypred)
    df_cm = pd.DataFrame(cf, columns=np.unique(y_true), index = np.unique(y_true))
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm/np.sum(df_cm), annot=True, fmt='.2%', vmin=0, vmax=1,)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted value')
    plt.ylabel('Real value')
    plt.show()

"""
## Model creation and prediction

We will first start with the naive bayes classifier which comes from a family of simple "probabilistic classifiers" based on application of Bayes theroem with strong independent assumptions between features.

The model is highly scalable, with number of parameters being linear with number of variables.
"""

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Train the model - CountVectorizer model
nb.fit(X_train_resample, y_train_resample)

# Class and probability predictions
yp_class = nb.predict(x_test_num)
yp_prob = nb.predict_proba(x_test_num)[:, 1]

from sklearn import metrics
print('Naive Bayes 1 by 2 Gram Spam Detection Model')
print("Accuracy", metrics.accuracy_score(y_test, yp_class))
# seaborn_conf(y_test, yp_class)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ["Not-Spam", "Spam"]
cf = confusion_matrix(y_test, yp_class)
make_confusion_matrix(cf, figsize=(10,7), cbar=False, group_names=labels, categories=categories)

from sklearn.metrics import classification_report
print('Naive Bayes Result\n')
print(classification_report(y_test, yp_class))

metrics.roc_auc_score(y_test, yp_prob)

import pickle

pickle.dump(nb, open('model.pkl', 'wb'))

"""Naive Bayes + Tfidf"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', MultinomialNB())])

pipe.fit(x_train, y_train)
yp_class = pipe.predict(x_test)
print(metrics.accuracy_score(y_test, yp_class))
seaborn_conf(y_test, yp_class)

"""XGBoost"""

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='auc'
    ))
    ]
)

pipe.fit(x_train, y_train)
yp_class_test = pipe.predict(x_test)
yp_class_train = pipe.predict(x_train)
yp_prob = pipe.predict_proba(x_test)[:,1]

print('Training accuracy score: {}'.format(metrics.accuracy_score(y_train, yp_class_train)))
print('Testing accuracy score: {}'.format(metrics.accuracy_score(y_test, yp_class_test)))

seaborn_conf(y_test, yp_class_test)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, yp_prob)

logit_roc_auc = metrics.roc_auc_score(y_test, yp_prob)
#print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test,yp_prob)
#print("fpr{}, tpr{}". format(fpr, tpr))

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, label='Naive Bayes (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC_NB.png')
plt.show()