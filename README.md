#### Attention Key Query and Value

The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will compute the similarity between your **query** (text in the search bar) against a set of **keys** (candidate video titles)  in their database, then present you the best matched videos (**values**).

**attention with one query vector**

suppose we have a query vector **q**, a list of key vectors $[k_1, k_2,...k_n]$, and a list of value vectors $[v_1, k_v,...v_n]$, $q \in R^{d}, k_i \in R^{d}, v_i \in R^{d}$ 

attention output by definition is a weighted average of value vectors
$$
\sum_{j}^{n}\alpha_{j}v_{j}
$$
where $\alpha_{j}$ are similarity scores and  $\sum \alpha_{j} = 1$. If we restrict $\alpha$ to be a one-hot vector (hard attention), this operation becomes the same as retrieving from a set of elements v with index $\alpha$. With the restriction removed (soft attention), the attention operation can be thought of as doing "proportional retrieval" according to the probability vector $\alpha$.

similarity scores $\alpha_{j}$ are computed through the similarity between the query q and the j-th key $k_j$
$$
\alpha_{j} = sim(q, k_j)
$$
usually, the similarity function is dot product with softmax. we take the dot product between q and each key to get the raw weights and apply softmax to make sure that the weights sum to 1.

the whole attention output given one query vector **q** is
$$
\sum_{j}^{n}\alpha_{j}v_{j} = sim(q, k_j) \cdot v_{j} = softmax(k_jq^T)\cdot v_{j}
$$
from the above definition, we know that the number of keys should be equal to the number of values

**attention with many query vectors**

suppose we have a list query vectors $[q_1, q_2,...q_m]$, we want to compute the attention output **for each** query vector. 

We need to reorganize the problem in matrix format: $Q \in R^{m \times d}, K \in R^{n \times d}, V \in R^{n \times d}$

we firstly compute the similarity weights:
$$
A = softmax(QK^T) \in R^{m \times n}
$$
there are m lines and n columns in **A**, each line represents the similarity scores for the corresponding query. softmax function is applied to make sure that weights in each line sum to one.

since attention output is a weighted average of value vectors, we can implement this by matrix multiplication
$$
Attention = AV \in R^{m \times d}
$$
for each query, we compute one attention output which is a weighted sum of values. Since we have m queries, we also have m attention output. 
$$
Attention(Q,K,V) = softmax(QK^T)V
$$
to sum up:

* the number of keys should be equal to the number of values

* we have m queries, we have m result. The result matrix shape is always consistent with the query matrix shape
* $Q \in R^{m \times d}, K \in R^{n \times d}, V \in R^{n \times d}, Result\  Attention\in R^{m \times d}$

#### Attention layer in encoder

Input: $x_e$ with shape $(B, dim, T_{in})$ from previous layer

both key, value, and query are  $x_e$, so it is also called self attention layer.

since the output shape should be same as query, which is $(B, dim, T_{in})$

the attention Mask is $(B, 1, T_{in}, T_{in})$

#### Attention in decoder

Input: $x_d$ with shape $(B, dim, T_{out})$ from previous layer
both key, value, and query are  $x_d$, so it is also called self attention layer.

since the output shape should be same as query, which is $(B, dim, T_{out})$

the attention Mask is $(B, 1, T_{out}, T_{out})$ 

#### Cross Attention in decoder

Input: $x_d$ with shape $(B, dim, T_{out})$ from previous layer, it can also access the feature map of the encoder which is also called **memory**, with shape $(B, dim, T_{in})$

Query: $x_d$

Keys and Values: **memory**

since the output shape should be same as query, which is $(B, dim, T_{out})$

the attention Mask $(B, 1, T_{in}, T_{out})$

