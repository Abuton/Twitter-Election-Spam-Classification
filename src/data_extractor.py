import ast, csv, sys
import validators
from datetime import datetime
import time

class Chen():

    def __init__(self):
        print("\n\n----- Starting Benevenuto Implementation ----")
        pass

    tweet_list = {}
    news_list = {}
    black_list = []

    spam_id = []
    def read_spam_word(self, filename):
        with open(filename) as words:
            for w in words:
                self.black_list.append(w.strip('\n'))

    def count_spam_word(self, tweet):
        s_count = 0
        for w in tweet.split():
            if w in self.black_list:
                s_count += 1
        return s_count


    def read_files(self, filename):
        # read tweet
        with open(filename, encoding = "ISO-8859-1") as tweet:
            t = csv.DictReader(tweet)
            try:
                i = 0
                for row in t:
                    self.tweet_list[i] = row
                    i += 1
            except csv.Error as e:
                sys.exit('Error in line %d: %s' % (t.line_num, e))

    def is_real_word(self, w):
        if validators.url(w):
            return False
        elif '#' in w:
            return False
        elif '@' in w:
            return False
        else:
            return True

    def count_words(self, tweet):
        w_count = 0
        for w in tweet.split():
            if self.is_real_word(w):
                w_count += 1

        return w_count

    def count_hashtag(self, tweet):
        if tweet['entities']['hashtags']:
            return len(tweet['entities']['hashtags'])
        return 0

    def count_urls(self, tweet):
        url_count = 0
        for w in tweet.split():
            if validators.url(w):
                url_count += 1
        return url_count

    def count_user_mentions(self, tweet):
        if tweet['entities']['user_mentions']:
            return len(tweet['entities']['user_mentions'])
        return 0

    def RT_count(self, tweet):
        rt_count = 0
        words = tweet.split()
        for w in words:
            if words[0] == 'RT' and '@' in w:
                rt_count += 1
        return rt_count

    def is_reply(self, tweet):
        w = tweet.split()
        if w[0] == 'RT':
            return True
        return False

    def count_character(self, tweet):
        char = 0
        num = 0
        if len(tweet) > 0:
            for w in tweet.split():
                if self.is_real_word(w):
                    char += sum(c.isalpha() for c in w)
                    num += sum(c.isdigit() for c in w)
            return char, num
        return char, num

    def generate_features(self, dataset):
        self.read_files(dataset.file_name)
        # account_age, tweet date minu account created date
        # no_of followers,
        # no_of followings,
        # no_userfavourites,
        # no_lists, and
        # no_of_tweets_by user,
        # number of numeric characters on the text,
        # number of characters that are numbers,
        # number of URLs,
        # number of hashtags,
        # number of mentions,
        # number of retweet

        wr = open(dataset.feature_file, 'w')
        wr = csv.writer(wr, dialect='excel')
        header = ['id', 'account_age', 'no_of_follower', 'no_of_following', 'no_user_fav', 'no_list', 'no_tweet', 'num_char', 'num_digit', 'num_url', 'num_hashtag', 'num_user_mention', 'num_rt',  'is_spam']
        wr.writerow(header)

        for row in self.tweet_list:
            if 1 == 1:# and row <= 75:
                feature_row = []
                feature_row.append(row)
                tweet = ast.literal_eval(self.tweet_list[row]['tweet'])
                tweet_text = tweet['text']
                #print("Tweet : ", tweet)

                FMT = '%Y-%m-%d %H:%M:%S'
                user_created_date = time.strftime(FMT, time.strptime(tweet['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                tweet_date = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                account_age = datetime.strptime(tweet_date, FMT)-datetime.strptime(user_created_date, FMT)
                #print("Account Age: ", account_age.days)

                no_of_follower = tweet['user']['followers_count']
                no_of_following = tweet['user']['friends_count']
                no_user_fav = tweet['user']['favourites_count']
                no_list = tweet['user']['listed_count']
                no_tweet = tweet['user']['statuses_count']

                hashtag_count = self.count_hashtag(tweet)
                url_count = self.count_urls(tweet_text)
                c_count, n_count = self.count_character(tweet_text)
                u_mention_count = self.count_user_mentions(tweet)

                num_rt = tweet['retweet_count']

                feature_row.append(account_age.days)
                feature_row.append(no_of_follower)
                feature_row.append(no_of_following)
                feature_row.append(no_user_fav)
                feature_row.append(no_list)
                feature_row.append(no_tweet)
                feature_row.append(c_count)
                feature_row.append(n_count)
                feature_row.append(url_count)
                feature_row.append(hashtag_count)
                feature_row.append(u_mention_count)
                feature_row.append(num_rt)


                if row in dataset.spam_id:
                    is_spam = True
                else:
                    is_spam = False

                feature_row.append(is_spam)
                wr.writerow(feature_row)
