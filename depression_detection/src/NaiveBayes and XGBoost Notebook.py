import warnings
warnings.filterwarnings('ignore')
import re
import string
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import pickle
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import itertools
import emoji
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Defining the global variables for the color schemes we will incorporate
pblue = "#496595"
pb2 = "#85a1c1"
pb3 = "#3f4d63"
pg = "#c6ccd8"
pb = "#202022"
pbg = "#f4f0ea"

pgreen = px.colors.qualitative.Plotly[2]


df = pd.read_csv('../data/tweets.csv')[['tweet', 'target']]
print(f' missing values count\n {df.isna().sum()}')
print('logest text length', np.max(df['tweet'].apply(lambda x: len(x.split())).values))

grouped_df = df.groupby('target').count().values.flatten()

fig = go.Figure()

fig.add_trace(go.Bar(
        x=['0'],
        y=[grouped_df[0]],
        name='Not-Depressed',
        text=[grouped_df[0]],
        textposition='auto',
        marker_color=pblue
))
fig.add_trace(go.Bar(
        x=['1'],
        y=[grouped_df[1]],
        name='Depressed',
        text=[grouped_df[1]],
        textposition='auto',
        marker_color=pg
))

fig.update_layout(
    title='Category distribution in the dataset')

fig.show()
# Creating series with length as index
# Sorting the series by index i.e. length
len_df_ham = df[df['target']=='0'].tweet.apply(lambda x: len(x.split())).value_counts().sort_index()
len_df_spam = df[df['target']=='1'].tweet.apply(lambda x: len(x.split())).value_counts().sort_index()


fig = go.Figure()
fig.add_trace(go.Scatter(
x=len_df_ham.index,
y=len_df_ham.values,
name='Not-Depressed',
fill='tozeroy',
marker_color=pblue))

fig.add_trace(go.Scatter(
x=len_df_spam.index,
y=len_df_spam.values,
name='Depressed',
fill='tozeroy',
marker_color=pg
))

fig.update_layout(
    title='Frequency of Tweets lengths')
fig.update_xaxes(range=[0, 80])
fig.show()

# emoticons
def load_dict_smileys():
    
    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

# self defined contractions
def load_dict_contractions():
    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }

def tweet_cleaning_for_sentiment_analysis(tweet):    
    
    #Special case not handled previously.
    tweet = tweet.replace('\x92',"'")
    #Removal of hastags/account
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    #Removal of address
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    #Removal of Punctuation
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    tweet = re.sub('\[.*?\]', '', tweet)
    tweet = re.sub('https?://\S+|www\.\S+', '', tweet)
    tweet = re.sub('<.*?>+', '', tweet)
    tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    tweet = re.sub('\n', '', tweet)
    tweet = re.sub('\w*\d\w*', '', tweet)
    #Lower case
    tweet = tweet.lower()
    #CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    CONTRACTIONS = load_dict_contractions()
    tweet = tweet.replace("’","'")
    words = tweet.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    tweet = " ".join(reformed)
    # Standardizing words
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    #Deal with emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons
    SMILEY = load_dict_smileys()  
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
    #Deal with emojis
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":"," ")
    tweet = ' '.join(tweet.split())

    return tweet

df['clean_tweet'] = df['tweet'].apply(tweet_cleaning_for_sentiment_analysis)

nltk.download('stopwords')

# Removing stop words
stop_words = stopwords.words('english')
more = ['u', 'im', 'c']
stop_words = stop_words + more


def sw_rem(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

df['clean_tweet'] = df['clean_tweet'].apply(sw_rem)

stems = nltk.SnowballStemmer('english')

def stemming(text):
    text = ' '.join(stems.stem(word) for word in text.split())
    return text

df['clean_tweet'] = df['clean_tweet'].apply(stemming)

x = df['clean_tweet']
y = df['target']

print(len(x), len(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=201)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

over = SMOTE(sampling_strategy='auto')
under = RandomUnderSampler(sampling_strategy='auto')

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# instantiate the vectorizer
count = CountVectorizer(stop_words='english', ngram_range=(1,2))
count.fit(x)


x_train_num = count.transform(x_train)
x_test_num = count.transform(x_test)

# transform the dataset
X_train_resample, y_train_resample = pipeline.fit_resample(x_train_num, y_train)

print(X_train_resample.shape)

# Working with TF-IDF now
from sklearn.feature_extraction.text import TfidfTransformer
# We are using transformer here
# If we use vectorizer, we can directly use the text
tfidf = TfidfTransformer()

tfidf.fit(X_train_resample)
x_train_tfidf = tfidf.transform(X_train_resample)
x_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Train the model - CountVectorizer model
nb.fit(X_train_resample, y_train_resample)

# Class and probability predictions
yp_class = nb.predict(x_test_num)
yp_prob = nb.predict_proba(x_test_num)[:, 1]

from sklearn import metrics
print('Naive Bayes 1 by 2 Gram Depression Detection Model')
print("Accuracy", metrics.accuracy_score(y_test, yp_class))
# seaborn_conf(y_test, yp_class)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ["Not-Depressed", "Depressed"]

x_axes = ['Not-Depressed','Depressed']
y_axes = ['Depressed', 'Not-Depressed']


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

from sklearn.metrics import confusion_matrix, roc_curve
categories=['Not-Depressed', 'Depressed']

def seaborn_conf(y, ypred):
    y_true = ["Not-Depressed", "Depressed"]
    y_pred = ["Not-Depressed", "Depressed"]

    cf = confusion_matrix(y, ypred)
    df_cm = pd.DataFrame(cf, columns=np.unique(y_true), index = np.unique(y_true))
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm/np.sum(df_cm), annot=True, fmt='.2%', vmin=0, vmax=1,)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted value')
    plt.ylabel('Real value')
    plt.show()


cf = confusion_matrix(y_test, yp_class)
make_confusion_matrix(cf, figsize=(10,7), cbar=False, group_names=labels, categories=categories, 
                        title='Naive Bayes Model')
print('')

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

pickle.dump(pipe, open('model_pipeline.pkl', 'wb'))

yp_class_train = pipe.predict(x_train)
yp_prob = pipe.predict_proba(x_test)[:,1]

print('Training accuracy score: {}'.format(metrics.accuracy_score(y_train, yp_class_train)))
print('Testing accuracy score: {}'.format(metrics.accuracy_score(y_test, yp_class_test)))

seaborn_conf(y_test, yp_class_test)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, yp_prob)

logit_roc_auc = metrics.roc_auc_score(y_test, yp_prob)
#print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test,yp_prob)


cf = confusion_matrix(y_test, yp_class_test)
make_confusion_matrix(cf, figsize=(10,7), cbar=False, group_names=labels, categories=categories)
print('XGBoost Model Metrics')

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, label='XGBoost Model (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC_XGBoost.png')
plt.show()
