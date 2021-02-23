## Word and Sentence Embeddings

- First order co-occurrences

  *syntagmatic(象征性的) associates* / relatedness (bee and honey)

- Second order co-occurrences
  *paradigmatic(范式的) parallels* / similarity (bee and bumblebee)

#### Distributional Semantics

Count words co-occur in a small sliding window, if the word occur, then the counter plus 1.

![embeddings]( https://imgur.com/download/4LPtnkI )

example: honey often co-occur with bee (First Order)

Calculate the similarity to figure out that bee and bumblebee are interchangeable (Second Order)

**Problems**:

It can be biased because of too popular words

**Solution**: Pointwise Mutual Information

Put the individual counts of the words to the denominator -- figure out they are randomly co-occurrent or not
$$
PMI = \log \frac {p(u, v)} {p(u) p(v)} = \log \frac {n_{uv}n} {n_u n_v}
$$

*Problem*: `PMI` may be -∞ when u, v never co-occur

*Solution*:
$$
pPMI = \max (0,PMI)
$$

> You shall know a word by the company it keeps.
>
> -- Firth, 1957

**Computing distance between vectors?**: Long and sparse!

 ![conversion](https://imgur.com/download/f3IVVDO) 

**What is context**:

![Context]( https://imgur.com/download/RCJA6Kg )

### Vector Factorization

##### GloVe

def:  Matrix factorization of log-counts with respect to weighted squared loss 

 GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

##### skip-gram

#### Word2vec

**2 architectures**

- Continuous bag-of-words:
  $$
  p(w_i | w_{i-h}, ...w_{i+h})
  $$
  
-  Continuous Skip-gram
  $$
  p(w_{i-h}, ...w_{i+h} | w_i)
  $$

**2 ways to avoid SoftMax** (it's used to produce these probabilities)

- Negative sampling
- Hierarchical SoftMax

##### Evaluation: word similarities

How do we test *similar words* have *similar vectors*?

- By linguists
- Use human judgements
- Compare *Spearman's correlation* between two lists

##### Evaluation: word analogies

- In cognitive science well known as *relational similarity* (v.s. *attributional similarity*)

- a: a' is as b:b'

### Doc2vec

Distributed Memory:
$$
p(w_i | w_{i-h}, ..., w_{i+h}, d)
$$
Distributed Bag of Words:
$$
p(w_{i-h}, ..., w_{i+h}|d)
$$
**Evaluation**: document similarities

1 doc, one similar doc, 1 different doc

### Word analogies without magic: king - man + woman = queen

#### BATS dataset

- Inflectional morphology
- Derivational morphology
- Lexicographic semantics
- Encyclopedic semantics

#### Form Characters to sentence embeddings

separate the words that have prefixes change and not change the meanings into 2 groups (make them far)