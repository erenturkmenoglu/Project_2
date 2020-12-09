# Importing libraries
import praw
import pandas as pd
from praw.models import MoreComments
import datetime

today = datetime.date.today()

# Authenticating Reddit account
reddit = praw.Reddit(client_id='ynA1ypO4kMVwRA', client_secret='dtFwv4-mgMDeI9uPm0GPieCRaFn8Dw', user_agent='Reddit_NLP')

subreddit_entry = input('Please enter a subreddit :')

# Pulling hot posts
hot_posts = reddit.subreddit(subreddit_entry).hot(limit=1)
for post in hot_posts:
    print(f"Title     : {post.title} \nID Number : {post.id}")

submission = reddit.submission(id=post.id)

# Get comments of a subreddit 
wallstreetbets_daily_text = open(f"_{subreddit_entry}_{today}_daily_discussion.txt", "w")
for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    print(top_level_comment.body)
    wallstreetbets_daily_text.write(top_level_comment.body)
wallstreetbets_daily_text.close()

comments_text = pd.read_csv('_wallstreetbets_2020-12-07_daily_discussion.txt', error_bad_lines=False, header=None)

df = pd.DataFrame(comments_text)

df = df.drop(columns=[1])

nasdaq_tickers = pd.read_csv('nasdaq.csv')

data = pd.concat([nasdaq_tickers, df], axis=1, ignore_index=True)

#define function to iterate through column of NASDAQ tickers, and look for instances of the items in column "text"
def check_subset(text,Nasdaq_Ticker):
    s = []
    for q in Nasdaq_Ticker:
        for r in text:
            if ((str(q) in str(r))) & (len(str(q))>3):
                s.append([q,r])
    return s

ticker_in_comments = check_subset(data[2], data[0])

# Change to pd dataframe
cleaned_df = pd.DataFrame(ticker_in_comments)

# Specify column names
cleaned_df.columns = [
    'Ticker',
    'Text'
]

#Save as csv
#cleaned_df.to_csv(f'_cleaned_{subreddit_entry}_{today}_tickers.csv', index=False)

# Run VADER sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import pandas as pd
import numpy

nltk.download('vader_lexicon')
# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create a sentiment scores DataFrame
topic_sentiments = []

for comment in cleaned_df['Text']:
    try:
        text = comment
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        topic_sentiments.append({
            "hot_post": text,
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu
            
        })
        
    except AttributeError:
        pass
    
sentiments_df = pd.DataFrame(topic_sentiments)

# Printing the output
sentiments_df

df_ticker_text_sentiment = pd.concat([cleaned_df['Ticker'], sentiments_df], axis=1)

df_ticker_text_sentiment.to_csv(f'_{today}_{subreddit_entry}_final_df.csv')

# Import dependencies for word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Create a graphic wordcloud
text_for_wordcloud = str(ticker_in_comments)
cloud = WordCloud(background_color="rgba(255, 255, 255, 0)").generate(text_for_wordcloud)
plt.figure(figsize=(10, 10), facecolor='b')
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

cloud.to_file(f'_wordcloud_{today}_{subreddit_entry}.png')




