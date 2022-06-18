import os
import time
import random

import nltk
import tqdm
import stanza


DATA_DIR = './data'
NLP = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

# Hyperparameters
MAX_LEN = 50
LOWER = False
REVERSE = False


def make_vocab():
    pass


def process_sent(sent, lower=False, reverse=False):
    if reverse:
        sent = reversed(sent)
    processed_sent = []
    for token in sent:
        if lower:
            tok = token.text.lower()
        else:
            tok = token.text
        processed_sent.append(tok)
    return processed_sent


def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    file = 'data/star_trek_episode_summaries.csv'
    summaries = []
    with open(file, encoding='utf8') as f:
        for line in f:
            doc = NLP(line.strip().lstrip('"').rstrip('"'))
            for sent in doc.sentences:
                if len(sent.tokens) <= MAX_LEN:
                    summary = process_sent(sent.tokens, lower=LOWER, reverse=REVERSE)
                    summaries.append(summary)
    random.shuffle(summaries)
    return summaries
