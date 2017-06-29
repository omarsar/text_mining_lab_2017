Hello Everyone, 

Here is the list of packages needed for our Text Mining Lab Session scheduled for 6/29/2017 (2:00-5:00 p.m.)

#### Updates:
------------------
* I have uploaded some poster examples of some past students
* For the guys intereted in the slack community, send me your email to ellfae@gmail and I will provide an invite
* If you have any other questions or technical problems, feel free to stop by Idea Lab Delta 701. I will be more than happy to support you. 
* I may extend the python notebook based on the excellent questions you guys asked (e.g., more statistics, visuals, etc.)
* Lastly, good luck and enjoy your stay here. 


#### Software:
------------------

* Python 3 (coding will be done strictly using Python 3)
* Anaconda Environment (recommended but not mandatory) (https://www.continuum.io/downloads)
* Jupyter (http://jupyter.org/)
* Google's word2vec (Download the file... warning! it is really huge)(https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
* Gensim (https://radimrehurek.com/gensim/)
* Scikit Learn (http://scikit-learn.org/stable/) (get the latest version)
* Pandas (http://pandas.pydata.org/)
* Matplotlib (https://matplotlib.org/)
* NLTK (for stopwords) (http://www.nltk.org/)

#### Computing Resources:
-------------------
* Operating System: Preferably Linux or MacOS (Windows break but you can try it out)
* RAM: 4GB 
* Disk Space: 8GB (mostly to store word embeddings)


#### Test:
-------------------
Once you have installed all the necessary packages, you can test to see if everything is working by running the following python code:

```python
import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords
%matplotlib inline

```

If you have any further questions please feel free to contact me at ellfae@gmail.com

Have Fun,

Elvis Saravia (Text Mining TA)