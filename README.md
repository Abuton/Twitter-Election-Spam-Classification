# Twitter Spam Detection with a focus on Election

Social networking sites have become very popular in recent years. Users  use  them  to  find  new  friends,  updates  their  existing  friends  with  their  latest  
thoughts and activities. Among these sites, Twitter is the fastest growing site. Its popularity  also  attracts  many  spammers  to  infiltrate  legitimate  users’  accounts  with a large amount of spam messages. In this paper, we discuss some user-based and content-based features that are different between spammers and legitimate users.  Then,  we  use  these  features  to  facilitate  spam  detection.  Using  the  API  methods provided by Twitter, we crawled active Twitter users, their followers/following  information  and  their  most  recent  100  tweets.  Then,  we  evaluated  
our detection scheme based on the suggested user and content-based features. Our results  show  that  among  the  four  classifiers  we  evaluated,  the  Random Forest  
classifier  produces  the  best  results.  Our  spam  detector  can  achieve  95.7%  precision and 95.7% F-measure using the Random Forest classifier.

## File name definition

`src` : is a folder that contains all the utilities functions in python scripts that was used to preprocess the data as well as explore the data

`notebooks` : contains ipynb files that explores the data and clean up the data

`data` : is used to store some of the data that have been used
