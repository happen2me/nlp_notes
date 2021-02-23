# Week2

## All about counting

### Video 1

Chain rule + Markov Assumption:
$$
p(W) = p(w_1)p(w_2|w_1)p(w_3|w_2)...p(w_k|w_{k-1})
$$
significance of it: use 2-gram probabilities to predict the probability of a sentence.

problems with the formula:

- Some words are not likely to appear at the first position
- Only the possibility of the fixed k length sentences sum up to 1

Bigram language model(w<sub>0</sub> and w<sub>k+1</sub> will be fake tokens):
$$
p(W) = \prod_{i=1}^{k+1} p(w_i|w_{i-1})
$$
Estimate the probabilities:
$$
p(w_i|w_{i-1}) = \frac {c(w_{i-1}w_i)} {\sum _{w_i} c(w_{i-1}w_i)} = \frac {c(w_{i-1}w_i)} {c({w_{i-1}})}
$$

### Perplexity: is our model surprised with a real text?

N-gram model:
$$
p(W) = \prod_{i=1}^{k+1} p(w_i|w_{i-n+1}^{i-1})
$$
Log-likelihood maximization:
$$
\log p(W_{train}) = \sum _{i=1} ^{N+1} \log p(w_i | w_{i-n+1}^{i-1})  → \max
$$
Estimates for parameters:
$$
p(w_i | w_{i-n+1}^{i-1}) = \frac {c(w_{i-n+1}^i)} {\sum_{w_i} c(w_{i-n+1}^i)} = \frac {c(w_{i-n+1}^i)} {c(w_{i-n+1}^{i-1})}
$$
​		*N* is the length of the **train corpus** (all words concatenated)

How to choose the n: it depends on how much data you have:

- bigrams might not be enough
- 7-grams never happen
- generally, the more data you have, you tend to choose longer grams

#### Evaluate the model on test data

How to evaluate: 

- Extrinsic*(外在的)* evaluation: measure the quality of the final application (we usually lack time and resource to do this -- we have to evaluate it after we build the whole application)

- Intrinsic evaluation
  - Hold-out (text) perplexity*(困惑)*: we hold out some data to compute perplexity later (same as train split and test split)

What is perplexity:

- Likelihood
  $$
  L = p(W_{test}) = \prod _{i=1} ^{N+1} p(w_i | w_{i-n+1}^{i-1})
  $$

- Perplexity (it's like likelihood in the denominator*(分母)*)
  $$
  \mathcal{P} = p(W_{test})^{- \frac {1} {N}} = \frac {1} {\sqrt [N] {p(w_{test})}}
  $$
  The lower the perplexity is, the better!

#### Out-of-vocabulary words

If it didn't appear in the training data, the probability will be 0 and the perplexity will be infinite😱

Simple tactic with OOV:

1. Build a vocabulary (e.g. by word frequencies)
2. Substitute OOV words by \<UNK\> (both in train and test!)
3. Compute counts as usual for all tokens
4. Profit

But there are still problems:

- Example:
  Train: This is the house that Jack build
  Test: This is Jack
  $$
  p(Jack | is) = \frac {c(is\:Jack)} {c(is)} = 0
  \\
  p(W_{test}) = 0
  \\
  perplexity = \inf 😱
  $$

#### Laplacian Smoothing

**Idea**: 

- Pull some probability from frequent bigrams to infrequent ones (ideas of all smoothing techniques)

- Just add 1 to the counts, V is number of words in out vocabulary (add-one smoothing)
  $$
  \widehat{p}(w_i|w_{i-n+1}^{i-1}) = \frac {c(w_{i-n+1}^{i}) + 1} {c(w_{i-n+1}^{i-1})+V}
  $$

- 

$$
\widehat{p}(w_i|w_{i-n+1}^{i-1}) = \frac {c(w_{i-n+1}^i)+k} {c(w_{i-n+1}^{i-1})+Vk}
$$

#### Katz backoff

Problem:

- Longer n-grams are better, but data is not always enough

Idea:

- Try a longer n-gram and back off to shorter is needed
  p(tilde) and alpha are for normalization

$$
\widehat{p}(w_i|w_{i-n+1}^{i-1}) = 
  \begin{cases}
    \tilde{p}(w_i|w_{i-n+1}^{i-1})       & \quad \text{if } c(w_{i-n+1}^{i})>0\\
    \alpha (w_{i-n+1}^{i-1})\tilde{p}(w_i|w_{i-n+2}^{i-1})  & \quad \text{otherwise }
  \end{cases}
$$

#### Interpolation smoothing

**Idea**:

- Use a mixture of several n-gram models

- A example of trigram model:
  $$
  \widehat{p}(w_i|w_{i-2}w_{i-1}) = \lambda _{1}p(w_i|w_{i-2}w_{i-1}) + \lambda_{2}p(w_i|w_{i-1}) + \lambda_3p(w_i)
  \\
  \lambda_1 + \lambda_2 + \lambda_3 = 1 
  $$

- The weights are optimized on a test (dev) set
- Optionally they can also depend on the context

#### Absolute discounting

To what extent should we pull this probability mass?

**Idea**

- Let's compare the counts for bigrams in train and test sets

####  Kneser-Ney smoothing 

Some words are not common but can suits many context, some words are popular but can only paired with some certain words. (e.g. malt vs (Hong) Kong)

Let's concern about  the probability of the words proportional to how many different contexts can go just before the word

**Idea**

- The unigram distribution captures the word frequency

- We need to capture the diversity of contexts for the word
  $$
  \widehat{p}(w) \propto |\{x : c(x \: w) > 0\}|
  $$

- Probably, the most popular smoothing technique

N-gram models + Kneser-Ney smoothing is a strong baseline in Language Modeling.

### Sequence tagging with probabilistic models

**Problem**: Given a sequence of *tokens*, infer the most probable sequence of *labels* for these tokens.

**Example**:

- part of speech tagging
- named entity recognition

Approaches:

- Rule-based models
- Separate label classifiers for each token
- **Sequence models** (HMM, MEMM, CRF)
- Neural networks

#### PoS tagging with HMMs

**Notation**:

**x** = x<sub>1</sub>, ..., x<sub>T</sub> is a sequence of words (input)

**y** = y<sub>1</sub>,..., y<sub>T</sub> is a sequence of their tags (labels)

We need to find the most probable tag sequences given the sentences:
$$
\boldsymbol{y} = \text{argmax}_y p(y|x) = \text{argmax}_yp(x, y)
$$

#### Hidden Markov Model

$$
p(\text{x}, \text{y}) = p(\text{x}|\text{y})p(\text{y}) ≈ \prod _{t=1} ^{T} p(x_t|y_t)p(y_t|y_{t-1})
$$

x is the observable(given) variable and y is a hidden(to predict) variable

**Markov assumption**:
$$
p(\text{y}) ≈ \prod _{t=1} ^{T} p(y_t |y_{t-1})
$$
**Output independence**:
$$
p(\text{x}|\text{y}) ≈ \prod _{t=1} ^{T} p(y_t | y_{t-1})
$$
**Formal definition of HMM**:

A Hidden Markov Model is specified by:

1. The set S = s<sub>1</sub>, s<sub>2</sub>,..., s<sub>n</sub> of hidden states
2. The start state s<sub>0</sub>
3. The matrix A of transition probabilities: a<sub>ij</sub> = p(s<sub>j</sub>|s<sub>i</sub>)
4. The set O of possible visible outcomes (Given tag, give word)
5. The matrix B of output probabilities: b<sub>ik</sub>=p(o<sub>k</sub>|s<sub>i</sub>)

Parameters (only 2 matrix):

- State transfer matrix: (N+1)(N), the +1 is because there is a s<sub>0</sub>, which is a start state
- Output probabilities: N * |O|, |O| is number of output words

If we could see the tags in train set. we would count:
$$
a_{ij} = p(s_j|s_i) = \frac {c(s_i → s_j)} {c(s_i)}
\\
b_{ik} = p(o_k | s_i) = \frac {c(s_i → o_k)} {c(s_i)}
$$
The same in more formal terms (MLE):
$$
a_{ij} = p(s_j | s_i) = \frac {\sum _{t=1} ^{T} [y_{t-1}=s_i, y_t = s_j]} {\sum _{t=1}^T [y_t = s_i]}
$$
But many times, we don't see the tags

#### Baum-Welch algorithm (a sketch)

**E-step**: posterior probabilities*(后验概率)* for hidden variables:
$$
p(y_{t-1}=s_i, y_t = s_j)
$$
Can be effectively done with dynamic programming (forward-backward algorithm)

**M-step**: maximum likelihood updates for the parameters (b matrix is the same):
$$
a_{ij} = p(s_j | s_i) = \frac {\sum _{t=1} ^{T} p(y_{t-1}=s_i, y_t=s_j)} {\sum _{t=1}^{T}p(y_t = s_i)}
$$

#### Decoding problem:

Main formula for HMM:
$$
p(\text{x}, \text{y}) = \prod _{t=1} ^{T} p(y_t|y_{t-1})p(x_t|y_t)
$$
The first p is transition probabilities and the second is output probabilities

**problem**

What is the most probable sequence of hidden states?
$$
\text{y} = \text{argmax}_\text{y} p(\text{y}|\text{x}) = \text{argmax}_\text{y}p(\text{x}, \text{y}) 
$$
**Viterbi decoding**

Let ``Q``<sub>t,s</sub> be the most probable sequence of hidden states of length ``t`` that finishes in the state ``s`` and generates ``o``<sub>1</sub>, ..., ``o``<sub>t</sub>. Let ``q``<sub>t,s</sub> be the probability of this sequence. 

Then  ``q``<sub>t, s</sub> can be computed dynamically:
$$
q_{t,s} = \max_{s'} q_{t-1, s'} \cdot p(s|s') \cdot p(o_t|s)
$$
The first p is transition probability: from previous state s' to state s

The second p is the output probability: the probability of generate o when in sate s

![Vertibe](https://imgur.com/download/iqqFFUj/)

![]( https://imgur.com/download/VWhh3ei )

Remember the choices every time. Then do a back trace at last

Read more about it on Wikipedia: [维特比算法](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)

Here's also a YouTube video explaining it: [Viterbi Algorithm](https://www.youtube.com/watch?v=6JVqutwtzmo)

***For sequence tagging task***

**A recap on HMM**
$$
p(\text{x}, \text{y}) = \prod _{t=1} ^{T} p(y_t|y_{t-1})p(x_t|y_t)
$$
It is a generative model: it models the joint probabilities of x and y

![An example of how a route is selected.](https://imgur.com/download/XLpLTYz)

Every arrow is a probability (transition probability & output probability)

#### Maximum Entropy Markov Model (MEMM)

$$
p(\text{y|x}) = \prod _{t=1} ^{T} p(y_t|y_{t-1}, x_t)
$$

Think y as state. So it's current tag give previous tag, and current x

It is **discriminative**, it models the conditional probability of y given x. The tags are observable, we just need to produce the probability for generating y.

![Explain of MMEM](https://imgur.com/download/pHdv2Zn)
$$
p(y_t|y_{t-1}, x_t) = \frac {1} {Z_t(y_{t-1}, x_t)} \exp \left( \sum_{k=1}^K \theta_k f_k (y_t, y_{t-1}, x_t) \right)
$$
Theta_k is weight, f_k is weight, Z_t is normalization constant

#### Conditional Random Field (linear chain)

$$
p(\text{y|x}) = \frac {1} {Z(x)} \prod _{t=1} ^{T} \left( \sum _{k=1} ^K \theta_k f_k(y_t, y_{t-1}, x_t) \right)
$$

Compared to MEMM, the normalization Z(x) is moved out, it's more complicated to calculate Z.

![CRF](https://imgur.com/download/41hPfOq)

#### Feature engineering

**Label-observation features**
$$
f_k(y_t, y_{t-1}, x_t) = [y_t = y] g_m(x_t) \\
f_k(y_t, y_{t-1}, x_t) = [y_t = y] [y_{t-1} = y'] \\
f_k(y_t, y_{t-1}, x_t) = [y_t = y] [y_{t-1} = y'] g_m(x_t)
$$
[] is label, g is observation

#### Dependencies on input

Trick: Pretend the current input ``x``<sub>t</sub> contains not only the current word ``w``<sub>t</sub>, but also ``w``<sub>t-1</sub> and  ``w``<sub>t+1</sub> and build observation functions for them as well.

The model is discriminative, so we can use the whole input.

#### Resume of the lesson

**Probabilistic graphical models**:

- Hidden Markov Models (generative, directed)
- Maximum Entropy Markov Models (discriminative, directed)
- Conditional Random Field (discriminative, undirected)

**Tasks**:

- Training: fit parameters (Baum-Welch for HMM)
- Decoding: find the most probable tags (Viterbi for HMM)

**Practice**:

- Features engineering + black-box implementation

Referring to this [link](https://www.coursera.org/learn/language-processing/supplement/CROHj/probabilities-of-tag-sequences-in-hmms) to recap the probabilities of tag sequences in HMMs.

## Neural Networks approach

**Problem**: Predict the next word

How to generalize word representations better:

- Learn **distributed representations** for words
- Express probabilities of sequences in terms of these distributed representations and learn parameters

Representation:
$$
C ^{|V|× m}
$$
​	-- matrix of distributed word representations. |V| (rows) is vocabulary length, m is column

#### Probabilistic Neural Language Model

$$
p(w_i|w_{i-n+1}, ... w_{i-1}) = \frac {\exp(y_{w_i})} {\sum _{w ∈ V} \exp(y_w)}
$$

​	-- SoftMax over components of y
$$
y = b + Wx + U\tanh(d+Hx)
$$
​	-- Feed-forward NN with tons of parameters
$$
x = [C(w_{i-n+1}), ... C(w_{i-1})]^T
$$
​	-- Distributed representation of context words. Dimension: (n×m)^T = m×(n-1)

![Imgur](https://imgur.com/download/O5zdd1s) 

#### Log-Bilinear Language Model

- Has much less parameters and non-linear activations
- Measures similarity between word and the context

$$
p(w_i|w_{i-n+1}, ... w_{i-1}) = \frac {\exp(\widehat{r}^T r_{w_i} + b_{w_i})}{\sum _{w∈V} \exp(\widehat{r}^Tr_w + b_w)}
$$



Word Representation:
$$
r_{w_i} = C(w_i)^T
$$
Representation of context:
$$
\widehat{r} = \sum _{k=1} ^{n-1} W_kC(w_{i-k})^T
$$
​	-- W matrix is different according to positions in context

#### RNN

**Architecture**

- Use the current state output

- Apply a linear layer on top

- Do *SoftMax* to get probabilities

  ![model architecture]( https://imgur.com/download/C1WNu5I )

**How to train it**

![train](https://imgur.com/download/kL6BgR4)

**How do we use it to generate language**

*Greedy: argmax*

![greedy approach](https://imgur.com/download/iI1uUP5)

*beam search*

Keep in mind n sequences, continue them in diff ways: compare probabilities, stick to n sequences

