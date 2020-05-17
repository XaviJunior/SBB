import re
import string

from lime.lime_text import LimeTextExplainer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language='english')
punctuation = string.punctuation

train_url = "https://raw.githubusercontent.com/XaviJunior/SBB/master/project_2/Data/train.csv"
test_url = "https://raw.githubusercontent.com/XaviJunior/SBB/master/project_2/Data/test.csv"


def replace_elongated_word(word):
    """
    From https://github.com/ugis22/analysing_twitter/
    """
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
    """
    From https://github.com/ugis22/analysing_twitter/
    """
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
        tweet = re.sub(r"@\w+", "", tweet)  # remove Twitter usernames

    if remove_rt:
        tweet = re.sub(r"\bRT\b", "", tweet)  # remove "RT"
        
    if remove_elongated:
        tweet = replace_elongated_words(tweet)  # replace elongated words like "goooooooaaaaaal"
    
    if remove_punctuation:
        tweet = re.sub(r"[^a-zA-Z0-9';:\(\)]", " ", tweet)  # keep only alphanumerical + apostrophe + smiley characters
        
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
    nodisaster_doc_matrix = vectorizer.transform(df[df["target"] == 0].text)
    disaster_doc_matrix = vectorizer.transform(df[df["target"] == 1].text)
    nodisaster_tf = np.sum(nodisaster_doc_matrix, axis=0)
    disaster_tf = np.sum(disaster_doc_matrix, axis=0)
    nodisaster = np.squeeze(np.asarray(nodisaster_tf))
    disaster = np.squeeze(np.asarray(disaster_tf))
    term_freq_df = pd.DataFrame([nodisaster, disaster],columns=vectorizer.get_feature_names()).transpose()
    term_freq_df.columns = ["Nothing", "Disaster"]
    term_freq_df["TOTAL"] = term_freq_df["Nothing"] + term_freq_df["Disaster"]
    
    return term_freq_df


def tokenizer(tweet):
    tokens = nlp(tweet)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]  # stemmer.stem(word.text).strip()
    tokens = [word for word in tokens if word not in nlp.Defaults.stop_words and word not in punctuation]
    
    return tokens


def create_vectorizer(mode, ngram_range, remove_tokens):
    for word in remove_tokens:
        nlp.Defaults.stop_words.add(word)
        
    if mode == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)
    elif mode == "count":
        vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)
        
    return vectorizer


def lime_plot(classifier, vectorizer, tweet, num_features):
    explainer = LimeTextExplainer(class_names=[0,1])
    c = make_pipeline(vectorizer, classifier)
    
    exp = explainer.explain_instance(tweet, c.predict_proba, num_features=num_features)
    
    exp.show_in_notebook(text=True)

    
def get_predictions(classifier, vectorizer, to_fit, test_data, test_name, export):
    classifier.fit(to_fit[0], to_fit[1])
    
    to_predict = vectorizer.transform(test_data["text"].values.tolist())
    test_data["target"] = classifier.predict(to_predict)
    probabilities = classifier.predict_proba(to_predict)
    
    if export:
        test_data[["id", "target"]].to_csv(test_name + ".csv", index=False)
        print("Predictions exported to " + test_name + ".csv")
    
    return test_data, probabilities
    

def augment_dataset(fitted_classifier, vectorizer, cleaning_args,
                    test_data_with_predictions, probabilities,
                    threshold):
    df_original = pd.read_csv(train_url, encoding="utf-8")

    # Selection of new records based on probability
    indexes = []
    i = 0
    
    for prob in probabilities:
        if prob[0] > threshold or prob[0] < (1-threshold):
            indexes.append(i)
        i += 1
  
    # Augmentation
    augments = pd.DataFrame(columns=["id", "keyword", "location", "text", "target"])
    a = 0
    
    for i in indexes:
        augments.loc[a] = test_data_with_predictions.iloc[i,:]
        a += 1

    df_augm = df_original.append(augments)
    
    # Cleaning and vectorization of augmented dataset
    df_augm["text"] = df_augm["text"].apply(clean_tweet, cleaning_args)
    X_augm = vectorizer.fit_transform(df_augm["text"].values.tolist())
    y_augm = df_augm["target"].values.tolist()

    return X_augm, y_augm
