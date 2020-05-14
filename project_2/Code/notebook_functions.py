import re
import string

from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language='english')
punctuation = string.punctuation


def replace_elongated_word(word):
    # From https://github.com/ugis22/analysing_twitter/
    regex = r"(\w*)(\w+)\2(\w*)"
    repl = r"\1\2\3"   
    if wordnet.synsets(word):
        return word
    new_word = re.sub(regex, repl, word)
    if new_word != word:
        return replace_elongated_word(new_word)
    else:
        return new_word


def replace_elongated_words(row):
    # From https://github.com/ugis22/analysing_twitter/
    regexrep = r"(\w*)(\w+)(\2)(\w*)"
    words = ["".join(i) for i in re.findall(regexrep, row)]
    for word in words:
        if not wordnet.synsets(word):
            row = re.sub(word, replace_elongated_word(word), row)
    return row


def clean_tweet(tweet, split_hashtags=False, remove_numbers=False, remove_users=False, remove_rt=False, remove_elongated=False, remove_punctuation=False):
    tweet = re.sub(r" \w{1,3}\.{3,3} http\S{0,}", " ", tweet)  # remove truncated endings
    tweet = re.sub(r"http\S{0,}", " ", tweet)  # remove other URLs
    tweet = re.sub(r".Û.", "'", tweet)  # replace strange representation of apostrophe
    tweet = re.sub(r"\bx....\b", " ", tweet)  # remove hexadecimal characters
    tweet = re.sub(r'(?<!\w)([A-Z])\.', r'\1', tweet)  # remove periods in acronyms
    
    # replace HTML codes
    tweet = re.sub(r"&amp;", "&", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&gt;", ">", tweet)
    
    # correct some typos
    tweet = re.sub(r"\bNorf\b", "North", tweet)
    
    if split_hashtags:
        tweet = " ".join([a for a in re.split('(#[A-Z][a-z]+)',tweet) if a])  # split hashtags with ThisPattern
    
    if remove_numbers:
        tweet = re.sub(r"\b[0-9]+\b", "", tweet)  # remove tokens with numbers only
    
    if remove_users:
        tweet = re.sub(r"@[A-Za-z0-9_]+", "", tweet)  # remove Twitter usernames
    
    if remove_rt:
        tweet = re.sub(r"\bRT\b", "", tweet)  # remove "RT"
        
    if remove_elongated:
        tweet = replace_elongated_words(tweet)  # replace elongated words like "goooooooaaaaaal"
    
    if remove_punctuation:
        tweet = re.sub(r"[^a-zA-Z0-9']", " ", tweet)  # keep alphanumerical characters + apostrophe only

    tweet = re.sub(r"\s+|\t|\n", " ", tweet)  # remove all white spaces, tabs and newlines
    
    return tweet.strip()


def clean(df, df_test, args, export=False):    
    df["text"] = df["text"].apply(clean_tweet, args=args)
    df_test["text"] = df_test["text"].apply(clean_tweet, args=args)
    
    if export:
        df["text"].to_csv("tweets.csv", index=False)
    
    return df, df_test


def token_counter(df, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    vectorizer.fit(df["text"])
    NoCata_doc_matrix = vectorizer.transform(df[df["target"] == 0].text)
    Cata_doc_matrix = vectorizer.transform(df[df["target"] == 1].text)
    NoCata_tf = np.sum(NoCata_doc_matrix, axis=0)
    Cata_tf = np.sum(Cata_doc_matrix, axis=0)
    NoCata = np.squeeze(np.asarray(NoCata_tf))
    Cata = np.squeeze(np.asarray(Cata_tf))
    term_freq_df = pd.DataFrame([NoCata,Cata],columns=vectorizer.get_feature_names()).transpose()
    term_freq_df.columns = ["Nothing", "Disaster"]
    term_freq_df["TOTAL"] = term_freq_df["Nothing"] + term_freq_df["Disaster"]
    
    return term_freq_df


def tokenizer(tweet):
    tokens = nlp(tweet)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]  # stemmer.stem(word.text).strip()
    tokens = [word for word in tokens if word not in nlp.Defaults.stop_words and word not in punctuation]
    
    return tokens


def create_vectorizer(mode, ngram_range, remove):
    for word in remove:
        nlp.Defaults.stop_words.add(word)
        
    if mode == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)
    elif mode == "count":
        vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)
        
    return vectorizer
