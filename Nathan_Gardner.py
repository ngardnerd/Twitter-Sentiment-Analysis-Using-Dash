import dash
import dash_html_components as html
import dash_core_components as dcc
import base64
import tweepy
from tweepy import OAuthHandler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import seaborn as sns
import numpy as np
import pandas as pd
import re
from collections import Counter

# imports for dashboard
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import base64



#initializing vector
translator = Translator()

#keys and tokens
consumer_api_key = ''
consumer_api_secret = '' 
access_token = ''
access_token_secret =''

authorizer = OAuthHandler(consumer_api_key, consumer_api_secret)
authorizer.set_access_token(access_token, access_token_secret)
api = tweepy.API(authorizer ,timeout=15)

analyser = SentimentIntensityAnalyzer()


# Simple sentiment analysis
def sentiment_analyzer_scoress(sentence):
    score = analyser.polarity_scores(sentence)
    print(score['compound'])


# Sentiment analysis that classifies sentiments into positive, neutral, and negative
# and includes non english option
def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text    
    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 'Positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'Neutral'
    else:
        return 'Negative'
    
# function for returning a number of tweets for a certain user
def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

# functions for cleaning tweets for analysis
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst


# user input and utilizing previous functions
user_id = 'BarstoolTenn'
count=200
userinput = list_tweets(user_id, count)
userinput=clean_tweets(userinput)

# separating tweet from sentiment for use in plotly
sents = []
for tw in userinput:
    try:
        st = sentiment_analyzer_scores(tw)
        sents.append(st)
    except:
        sents.append(0)


f = {'Sentiment': ['Negative','Neutral','Positive'], 'Count': [sents.count('Negative'),sents.count('Neutral'),sents.count('Positive')]}
df = pd.DataFrame(data = f)



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import PIL
from PIL import Image

stopwords = set(STOPWORDS)
all_words = ' '.join([text for text in userinput])
wc = WordCloud(background_color="white", stopwords = stopwords, width = 500, height = 400, random_state=21,colormap = 'jet', max_words=100, max_font_size = 200)
wc.generate(all_words)
wc.to_file("userinput.png")

image_filename2 = 'userinput.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
image_filename = 'Twitter-Bird-Logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                     height = 75,width = 75)
            
        ], className = "one column", style={'textAlign': 'center'}),
        
        html.Div([
            html.H1('Twitter Sentiment Analysis')
        ], className='ten columns')
    ], className = 'row', style = {'textAlign': 'center'}),
    
    
    html.Div([
        html.Div([
            html.Label('Twitter Handle:'),
            dcc.Input(value = 'BarstoolTenn', id = 'userinput')], className = 'six columns', style={'textAlign': 'right'}),
        html.Div([
            html.Label('Number of Tweets: ', style = {'textAlign':'left'}),
            dcc.Input(value = '200', id = 'userinput2')], className = 'six columns', style={'textAlign':'left'})
    ], className = 'row', style = {'textAlign':'center'}),

    html.Div([
    html.Div([html.H4('Sentiment Visualization'),  
        dcc.Graph(id = 'sentimentGraph',
                                   figure = {
                                       'data':[
                                           {
                                               'x': df['Sentiment'],
                                               'y': df['Count'],
                                               'type': 'bar'
                                           }
                                       ],
                                       'layout' : go.Layout(
                                           xaxis = {'title': 'Tweets Sentiment'},
                                           yaxis = {'title': '#Tweets'},
                                           colorway = ['#55acee'],
                                           plot_bgcolor= 'white',
                                           paper_bgcolor = 'white')
                                   } )
    ],className = 'six columns'),
    html.Div([html.H4('Tweet Wordcloud'),
        html.Img(id = 'pic', src='data:image/png;base64,{}'.format(encoded_image2.decode()))
    ],className = 'six columns')
    ],className = 'row', style = {'textAlign': 'center'})
])

@app.callback(
    Output(component_id = 'sentimentGraph', component_property = 'figure'), 
    [Input(component_id = 'userinput', component_property = 'value'), Input(component_id = 'userinput2', component_property = 'value')])
def update_graph(option1, option2):
    # user input and utilizing previous functions
    user_id = option1
    count= option2
    userinput = list_tweets(user_id, count)
    userinput=clean_tweets(userinput)

    # separating tweet from sentiment for use in plotly
    sents = []
    for tw in userinput:
        try:
            st = sentiment_analyzer_scores(tw)
            sents.append(st)
        except:
            sents.append(0)


    f = {'Sentiment': ['Negative','Neutral','Positive'], 
         'Count': [sents.count('Negative'),sents.count('Neutral'),sents.count('Positive')]}
    df = pd.DataFrame(data = f)


    figure = {
                'data':[
                            {
                                'x': df['Sentiment'],
                                'y': df['Count'],
                                'type': 'bar'
                                }
                            ],
                             'layout' : go.Layout(
                                xaxis = {'title': 'Tweets Sentiment'},
                                yaxis = {'title': '#Tweets'},
                                colorway = ['#55acee'],
                                plot_bgcolor= 'white',
                                paper_bgcolor = 'white'),
                                   } 
    return figure

@app.callback(
    Output(component_id = 'pic', component_property = 'src'), 
    [Input(component_id = 'userinput', component_property = 'value'), Input(component_id = 'userinput2', component_property = 'value')])
def update_img(option1,option2):
    user_id = option1
    count= option2
    userinput = list_tweets(user_id, count)
    userinput=clean_tweets(userinput)
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in userinput])
    wc = WordCloud(background_color="white", 
    stopwords = stopwords, 
    width = 500, 
    height = 400,                         
    random_state=21,colormap = 'jet', 
    max_words=100, 
    max_font_size = 200)
    wc.generate(all_words)
    wc.to_file("userinput.png")

    image_filename2 = 'userinput.png' # replace with your own image
    encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
    
    src='data:image/png;base64,{}'.format(encoded_image2.decode())
    return src



if __name__ == '__main__':
    app.run_server(debug=False)

    
sentiment_analyzer_scoress("I LOVE learning python!")
