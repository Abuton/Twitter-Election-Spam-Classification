# import the module
import tweepy
from pathlib import Path 
import json
import csv

def get_twauth():
    path = Path('.') / '.env/twitter_keys.json'
    twitter_auth = json.load(open(path))
    return twitter_auth

tw_auth = get_twauth()

  
# assign the values accordingly
consumer_key = tw_auth['TWITTER_CONSUMER_KEY']
consumer_secret = tw_auth['TWITTER_CONSUMER_SECRET']
access_token = tw_auth['TWITTER_ACCESS_TOKEN']
access_token_secret = tw_auth['TWITTER_ACCESS_TOKEN_SECRET']


# authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  
# set access to user's access key and access secret 
auth.set_access_token(access_token, access_token_secret)
  
# calling the api 
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

if __name__ == "__main__":
    twitter_screen_name = ['benpolitico', 'daveweigel', 'fixfelicia', 'SusanPage', 'alex_wags', 'LarrySabato', 'cbellantoni', 'stevebenen', 'Atrios',
                      'fivethirtyeight', 'RasmussenPoll', 'ggreenwald', 'mikememoli', 'LamidiKehinde8', 'dmarinere', 'salisolar', 'geeksforgeeks', 'nicopitney',
                      'aproko_doctor', 'aycomdian']
    
    # getting only 50 followers
    screen_name_twitter = []
    for screen_name in twitter_screen_name[:10]:
        try:
            for follower in tweepy.Cursor(api.followers, screen_name=screen_name).items(2):
                screen_name_twitter.append(follower.screen_name)
        except tweepy.TweepError:
            print(screen_name)
            
    column_name = "ScreenName"
    twitter_screen_name.append(screen_name)
            
    with open('screenName.csv', 'w', newline='') as f:
      
        # using csv.writer method from CSV package
        write = csv.writer(f, quoting=csv.QUOTE_ALL)

        write.writerow(twitter_screen_name)

