# Blog for: Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

## Intro
- Brief overview of the paper
- what it attempts to do
- Their results
- List out what we will do 
## Reproducing their results
## Porting to pytorch
## Positional Encoding:

Transformers do not take the position of its input data points into account. They treat each point as independent. But the order of the points in a curve are highly relevant, therefore we use positional encoding to give information on the order of the input sequence to the transformer. This is done by mapping each point of the curve to a positional vector. There exists many variants of positional encoding.

The paper implements 3 types of positional encodings, the standard sin-cos positional encoding, learned positional encoding and paired scrambled positional encoding. Sine-Cosine positional encoding is the most widespread technique. It encodes each point by alternatively applying these functions along the position vector. Intuitively, each point is mapped to a section of the sin/cos curve, giving it a value between -1 and 1. The frequency is adjusted to match the input length so that no two points have the same value.
$$P(k,2i) =  sin(\frac{k}{n^{2i/d}})$$
$$P(k,2i + 1) =  cos(\frac{k}{n^{2i/d}})$$
- K refers to the value of the point on the x-axis, its position in the input  sequence.
- d is the length of the vectors we are encoding the points into (the number of columns in the encoding matrix). 
- n = [0,99], it is the length of the input curve
- i indexes the columns of the encoding matrix for both sin and cos columns. 0 <= i <= d/2
Figure 1 shows an example computation of the positional encoding matrix.

![pos_enc](https://github.com/Remi-Lejeune/deep-learning-project/assets/72941971/70a6e302-4e3c-4022-af1f-aa2f13c2f850)

The second positional encoding used by the paper is learned positional encoding. Instead of using a fixed mathematical model and equations, the positional encodings are learned with the model parameters during training. This allows for a more tailored embedding as it can be modeled to specific characteristics of the data. 

The third and final encoding they use is paired scrambled positional encoding. This seems to be a custom implementation of the paper as we could not find other references to it. It inherits from the standard sin-cosine encoding mentioned above but goes on  to group the embeddings into pairs, permuting them randomly before returning. Thus retaining some pairwise positional information but losing information on the total sequence. We are unsure of  the direct use of this encoding though it might relate to regularizing or testing the generalization ability of the transformer model.

For our code variant we decided to implement a custom encoding. We created a positional encoding module based on the euclidean distance of the points position (x-value) to the start of the curve. We wanted to see if a simple model could achieve the same results as the more complex ones mentioned above.

|   | LCPFN | Large model | Small model | Small model (No forced teaching) | Small  model (No  forced teaching) (Positional encoding: Euclidian distance)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Mean  |  **0.0002**  | 2.3589 | 3.2686 | 1.9210 | 3.4749 |
| Standard Deviation  | **8.0023e-05**  | 0.3185 | 0.5678 | 0.2005 | 0.5448 |


- Result and differences 
- why?



## Testing their model on new data
## Limitations
- Each makes a paragraph of our limitations
## Conculsion and future work
