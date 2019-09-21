import datetime
from random import randint, choice
import re

from pymongo import MongoClient

import nltk
from nltk.corpus import stopwords

from django.shortcuts import render, redirect, render_to_response


_USER = 'MONGO USERNAME'
_PASS = 'MONGO PASSWORD'
_IP = 'MONGO IP'
_PORT = 'MONGO PORT'

mongodb = MongoClient(f'mongodb://{_USER}:{_PASS}@{_IP}:{_PORT}')

sentiment_documents = mongodb['sentiment']['streaming_sentiment_tweets']
classified_sentiments = mongodb['sentiment']['supervised_sentiment_tweets']

NO_REPLY_TWEETS = sentiment_documents.find({'in_reply_to_user_id': None}).count()

nltk.download('stopwords')
sw_spanish = stopwords.words('spanish')

global sentiment_tweets
sentiment_tweets = list()


def index(request):
    with open('log.txt', 'a') as f:
        f.write(str(get_client_ip(request)) + ' ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
    f.close()

    for document in sentiment_documents.find({'in_reply_to_user_id': None}).skip(randint(1, NO_REPLY_TWEETS - 10)).limit(10):
        if 'extended_tweet' in document:
            obj = {
                'id_str': document['id_str'],
                'text': document['extended_tweet']['full_text']
            }

            sentiment_tweets.append(obj)
        else:
            obj = {
                'id_str': document['id_str'],
                'text': document['text']
            }

            sentiment_tweets.append(obj)

    return render_to_response('index.html')


def sentiment_classification(request):
    with open('log.txt', 'a') as f:
        f.write(str(get_client_ip(request)) + ' ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
    f.close()

    if len(sentiment_tweets) == 0:
        while len(sentiment_tweets) == 0:
            sentiment_tweets.clear()

            for document in sentiment_documents.find({'in_reply_to_user_id': None}).skip(randint(1, NO_REPLY_TWEETS - 10)).limit(10):
                if 'extended_tweet' in document:
                    obj = {
                        'id_str': document['id_str'],
                        'text': document['extended_tweet']['full_text']
                    }

                    sentiment_tweets.append(obj)
                else:
                    obj = {
                        'id_str': document['id_str'],
                        'text': document['text']
                    }

                    sentiment_tweets.append(obj)

    print(sentiment_tweets)

    tweet = choice(sentiment_tweets)

    print(tweet)

    request.session['original_tweet'] = tweet['text']
    request.session['id_str'] = tweet['id_str']

    reg = re.sub('[^[a-zA-Z ?¡¿!.,à-úÀ-Ú]|https?://.*[\r\n]*|(@[A-Za-z0-9]+)|]', '', tweet['text']).strip()
    reg = re.sub(' +', ' ', reg)

    request.session['tweet'] = reg

    context = {
        'text': reg,
        'original_text': tweet['text'],
    }

    return render(request, 'sentiment_supervised.html', context)


def sentiment_positive(request):
    tweet = request.session.get('original_tweet')
    id_str = request.session.get('id_str')

    obj = {
        'id_str': id_str,
        'text': tweet
    }

    sentiment_tweets.remove(obj)

    if classified_sentiments.find({'tweet': request.session.get('tweet')}).count() > 0:
        response = redirect('/classification')
        return response
    else:
        obj = {
            'tweet': request.session.get('tweet'),
            'sentiment': 'positive',
        }

        classified_sentiments.insert_one(obj)

        query = {
            'id_str': id_str
        }

        sentiment_documents.delete_one(query)

        response = redirect('/classification')
        return response


def sentiment_neutral(request):
    tweet = request.session.get('original_tweet')
    id_str = request.session.get('id_str')

    obj = {
        'id_str': id_str,
        'text': tweet
    }

    sentiment_tweets.remove(obj)

    if classified_sentiments.find({'tweet': request.session.get('tweet')}).count() > 0:
        response = redirect('/classification')
        return response
    else:
        obj = {
            'tweet': request.session.get('tweet'),
            'sentiment': 'positive',
        }

        classified_sentiments.insert_one(obj)

        query = {
            'id_str': id_str
        }

        sentiment_documents.delete_one(query)

        response = redirect('/classification')
        return response


def sentiment_negative(request):
    tweet = request.session.get('original_tweet')
    id_str = request.session.get('id_str')

    obj = {
        'id_str': id_str,
        'text': tweet
    }

    sentiment_tweets.remove(obj)

    if classified_sentiments.find({'tweet': request.session.get('tweet')}).count() > 0:
        response = redirect('/classification')
        return response
    else:
        obj = {
            'tweet': request.session.get('tweet'),
            'sentiment': 'positive',
        }

        classified_sentiments.insert_one(obj)

        query = {
            'id_str': id_str
        }

        sentiment_documents.delete_one(query)

        response = redirect('/classification')
        return response


def sentiment_none(request):
    tweet = request.session.get('original_tweet')
    id_str = request.session.get('id_str')

    obj = {
        'id_str': id_str,
        'text': tweet
    }

    sentiment_tweets.remove(obj)

    response = redirect('/classification')
    return response


def sentiment_refresh(request):
    sentiment_tweets.clear()

    for document in sentiment_documents.find({'in_reply_to_user_id': None}).skip(randint(1, NO_REPLY_TWEETS - 10)).limit(10):
        if 'extended_tweet' in document:
            obj = {
                'id_str': document['id_str'],
                'text': document['extended_tweet']['full_text']
            }

            sentiment_tweets.append(obj)
        else:
            obj = {
                'id_str': document['id_str'],
                'text': document['text']
            }

            sentiment_tweets.append(obj)

    response = redirect('/classification')
    return response


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')

    return ip
