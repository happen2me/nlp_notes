## Text Preprocess

### Tokenization 

**split input sequence into different tokens**

**token**: a semantic unit (can be word, phrases, ...). *we can make our own rules*
with python:

```python
tokenizer = nltk.tokenize.WordPunctTokenizer()
tokenizer.tokenzie(text)
```

### Token Normalization

**stemming**: remove suffix,  use *heuristics* rules

```

```



**lemmatization**: use database to find lemma form.
might not merge same word (talked->talked, talking->talking)

**Further Normalization**

- Capital letter, acronyms



## Feature Extraction

#### Bag of words

| good | movie | not  | a    | did  | like |
| ---- | ----- | ---- | ---- | ---- | ---- |
| 1    | 1     | 0    | 0    | 0    | 0    |

**Problems**

- lose order
- counters are not normalized

**n-grams** (to preserve order)

Example: 1-grams for token, 2-grams for token pairs

| good movie | movie | did not | a    |
| ---------- | ----- | ------- | ---- |
| 1          | 1     | 0       | 0    |
| 1          | 1     | 0       | 1    |

​	problems: too many feature (solution as below)

**Remove some n-gram**

- ~~high frequency n-grams: stop words~~
- ~~low frequency n-grams: typos~~
- medium frequency n-grams: we'll use these grams √

**Medium frequency n-grams** (still too many)

- n-gram with smaller frequency can be more **discriminating** because it can capture specific issue

**Term Frequency (TF)**: frequency for `t` in doc `d`

- count_t / total_count_of_all
- log form: 1 + log(f_t,d)

#### TF-IDF

**Inverse Document Frequency**
$$
idf(t, D) = log\frac{N}{|d∈D:t∈d|}
$$
**TF-IDF**
$$
tf-idf(t, d, D) = tf(t,d) * idf(t, D)
$$
**important** = specific issue in a document that is not frequent in the whole collection of documents

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import panda as pd
texts = [
    "good movie", "not a good movie", "did not like", 
    "i like it", "good one"
]
tfidf = TfidfVetorizer(min_df=2, max_df=0.5, ngram_range=(1, 2)) #min_df: throw low fq, max_df: throw stop words
features = tfidf.fit_transform(texts)
pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)
```

#### Better BOW

replace counters with TF-IDF values

## Linear models for sentiment analysis

for bag of words representation: 

- Linear model:   fine and fast and works for millions of features. 
- Naïve Bayes:  fine and fast and works for millions of features. 

A model for sentiment classification: Logistic regression

- p(y=1|x) = σ(ω^T * x)
- can handle sparse data
- easy to interpret

**Make it better**:

- tokenization: like ":)"
- normalization: stemming or lemmatization
- different models: SVM, Naïve Bayes
- throw BOW and use deep learning

## Spam Filter

### Mapping n-grams to feature indices

using hash map has problem in case of large corpus (1TB)

### Hashing

**Use  hashing**
$$
n-gram => hash(n-gram) \% 2^{b}
$$

```python
sklearn.feature_extraction.text.HashVectorizer
```

**Personalized Hashing**: have feature for particular user with personalized preference
$$
ф(token) = hash(u + "\_" + token)
$$
example: user123_like

reason: 

- some user think newsletter as spam, but most of others don't
- (even new users have better performance) we learn better "global" preference having personalized features which learn "local" user preference. Like we only help those hate newsletter with that filter, while the MAJORITY feel ok with newsletter

## Neural networks for words

neural needs **dense** representation

**word2vec** property:

words with similar context tend to have collinear vectors

summation of word2vec already performs well as text descreptors



#### A better way: 1D convolution

![avatar](https://imgur.com/download/SJl9WGd)

- the convolution provide high activation for certain meaning
- word2vecs of similar meaning have close distance
- can be extended to 3-grams, 4-grams, etc.
- One filter is not enough, need to track many n-grams
- the name !D came from the 1-direction sliding

**Final Architecture**

- 3,4,5-gram windows with 100 filters each

- take the maximum value as the output of a convolution operation

- MLP(Multi-Layer positron) on top of 300 features 

  

### Neural Networks for characters

think of text as sequence of characters

use one-hot encoded characters 

( Don't forget to normalize these features row-wise! )

**Procedure**

1. use n-gram filters to convolute over characters

   Filter#1: 0.4|0.8|0.5|....

   Filter#2: 0.5|0.6|0.4|....

2. pooling output: take neighbor values and take maximum
   **reason**: pooling provide a little bit of position invariance

3. stack more layers ( We cannot apply linear models right now, we have a variable length of that representation and our features are too weak, we've looked only on character n-grams! )
   the length of feature length decrease

4. repeat it for some times (6 here)