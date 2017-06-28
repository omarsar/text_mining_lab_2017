import nltk
import numpy as np
import gensim

""" Helpers for Word2vec """

def word_averaging(wv, words):
    """
    Word Averaging for a specific set of words
    """
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    """
    Return word averaging list
    """
    return np.vstack([word_averaging(wv, review) for review in text_list ])

def w2v_tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using word2vec
    """
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if remove_stopwords == True:
                if word in stopwords.words('english'):
                    continue
            tokens.append(word)
    return tokens

"""
Visualization functions
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def unpack_words_from_doc_vector(model):
    words_found = []
    for x in zip(model.wv.vocab):
        words_found.append(x[0])
    return words_found

def visualize_vectors(model, words):
    X = model[model.wv.vocab]
    
    tsne = TSNE(n_components = 2)
    X_tsne = tsne.fit_transform(X[:1000,:])
    
    plt.scatter(X_tsne[:300, 0], X_tsne[:300, 1])
    
    # visualize
    for label, x, y in zip(words[:300], X_tsne[:300, 0], X_tsne[:300, 1]):
        plt.annotate(label, xy=(x,y), xytext=(0,0),  textcoords='offset points')
    plt.show