import re
import nltk
import string
from nltk.corpus import stopwords
import pickle

stems = nltk.SnowballStemmer('english')
nltk.download('stopwords')

# Removing stop words
stop_words = stopwords.words('english')


def cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def pipeline(text):
    text = cleaning(text)
    text = ' ' .join(word for word in text.split(' ') if word not in stop_words)
    text = ' '.join(stems.stem(word) for word in text.split(' '))
    return text

sentence = "A former Deputy Governor 's Central Bank said million Nigerians"
sentence1 = 'URGENT! We are trying to contact U. Todays draw shows that you have won a å£2000 prize GUARANTEED. Call 09066358361 from land line. Claim Y87. Valid 12hrs only,'

sentence_transform = pipeline(sentence)
sentence1_transform = pipeline(sentence1)

model = pickle.load(open('models/model_pipeline.pkl', 'rb'))

prediction = model.predict([sentence1_transform])
prediction_proba = model.predict_proba([sentence1_transform])
if prediction[0] == 0: 
    prediction = 'Non Spam Tweet'
else: 
    prediction = 'Spam Tweet'

print(f"{sentence1} is a {prediction}")
print(f"Model is {round(prediction_proba.max(), 2)}% Acuurate")