# Blog for: Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

## Introduction
In the machine learning field, where computational requirements are fast increasing with the needs for more powerful predictive models, being able to predict a model’s training performance when given more resources would be of great benefit for training efficiency both time and cost-wise.

The paper "Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks" [1] addresses this issue by innovating within the domain of learning curve extrapolation by employing a Bayesian approach tailored to handle the intrinsic uncertainties of predicting machine learning model performance over the training process. Traditional methods, while effective, often pose limitations due to their computational expense or restrictive assumptions on the data. This research offers an alternative by utilizing Prior-Data Fitted Networks (PFNs) for learnign curve inference – dubbed Learning Curve Prior-Data Fitted Networks (LC-PFNs), which leverage transformer architectures to perform Bayesian inference efficiently and flexibly.

Building on the work presented in [2], who introduced a parametric model for learning curve extrapolation using Markov Chain Monte Carlo (MCMC) methods, this paper advances the field by dramatically enhancing computational efficiency. Notably, LC-PFNs can deliver at leaset equally accurate extrapolations more than 10,000 times faster than traditional MCMC methods, providing a potent tool for applications like automated machine learning and hyperparameter optimization where quick, reliable forecasts can significantly accelerate decision-making processes. Figure 1, obtained from [1] compares the reported performance in terms of log-likelyhood vs. Inference runtime.
Naturally, as the authors point out, this increase in performance comes at the cost of a one-time significant overhead of traininf the LC-PFN.
![Example Image](images/original_performance.png)

In their methodology, the authors initially develop and test their proposed model using a set of informative priors (obtained from building upon Dohman uninformative priors) to sample learning curves that are used for training and inference. Subsequently, the LC-PFN model is rigorously tested with real-world data and evaluated for its effectiveness in predictive early stopping based on its forecasts. In almost all cases, the LC-PFN model matches or exceeds the accuracy of the MCMC model, while performing the inference step thousands of times faster.

In this blog post, an initial reproduction of the obtained results will be attempted using the authors’ original code as faithfully as possible. Following the replication, the focus will shift to adapting the original codebase to leverage the tools provided by PyTorch, particularly the Transformer class. Notably, in [1] the authors develop a custom and quite convoluted implementation of the transformer model. By porting the model to utilize PyTorch’s built-in Transformer modules, the expectation is to streamline the architecture and potentially enhance computational efficiency and maintainability of the code. Finally, the original implementation will be put to further testing using an increased set of real-world data.

## Reproducing the obtained results

The initial goal was to replicate the learning curve inference demonstrated by both the LC-PFN and the MCMC models using the authors' original code with minimal changes. Success was defined by the ability to recreate an image similar to Figure 2, obtained from the original paper [1], showing the inference from both models of two curves with different cutoffs at 10 and 20.
![Example Image](images/original_inference.png)


The hardware used in this section and the equivalent used in the original paper is listed in Table 1.
|                    | LC-PFN Training and inference | MCMC fitting and inference    |
|--------------------|-------------------------------|-------------------------------|
| Original paper     | single RTX2080 GPU            | single Intel(R) Xeon(R) Gold 6242 CPU |
| Reproduction       | Single T4 GPU                 | single Intel(R) Core(TM) i7-8550U CPU |

*Table 1:  Hardware used in the original paper [1] and in the reproduction*


### Inference with LC-PFN

The first step was to obtain a trained LC-PFN model. To this end, the interface provided by the authors for trying such models was used.

The model was set to handle sequences of length 100 and included an embedding size of 256 and 3 layers. This architecture was chosen to mimic that of the smaller model tested by the authors, thus testing if the least powerful model highlighted in the paper is indeed able to achieve satisfactory results. The model was trained with a batch size of 500 for 300 epochs (a total training time of slightly less than five hours compared to the total of around 8 hours mentioned in the paper for the largest model).

After training the model, it was put to the test with two target curves shown with cutoffs 10 and 20 as previously mentioned. The target curves and the network’s predictions when fed 10 or 20 epochs of each curve are shown in Figure X. The result is, indeed, quite satisfactory and as such the reproduction of the LC-PFN pipeline from training to inference using the authors’ code is considered successful.
![Example Image](images/obtained_inference.png)


### Inference with MCMC

Moving on to the reproduction of the inference capabilities of the MCMC model, again, the code provided by the author’s was analyzed in order to find the correct pipeline for using this model. Unlike in the case of the LC-PFN, however, there is no “training process” in the sense that the term used in deep learning. Instead, each model is obtained by fitting it to a portion (up to the cutoff) of the sequence to be predicted.

The reproduction process, unlike that of the LC-PFN, was unsuccessful. This was due to the sheer amount of runtime required by the MCMC model. Of course, the main objective of the paper is to demonstrate that the prediction runtime of the LC-PFN is much faster than that of the MCMC model, however, the required runtime observed during reproduction was incomparably larger than that reported by the authors. Take as an example Table 2 which lists the runtimes reported by the authors as well as that observed in the reproduction, where the mentioned discrepancy is clearly visible.
| Source        | Parameters                                   | Avg. Time (seconds) |
|---------------|----------------------------------------------|---------------------|
| Original Paper| nsamples = 2000, nwalkers = 100, burn-in = 500, thin = 1 | 54.401 |
| Original Paper| nsamples = 4000, nwalkers = 100, burn-in = 100, thin = 100 | 45.160 |
| Original Paper| nsamples = 4000, nwalkers = 100, burn-in = 500, thin = 1 | 103.151 |
| Original Paper| Nsamples = 100, nwalkers = 100, burn-in and thin unknown | ~1 |
| Reproduction  | Nsamples = 100, nwalkers = 100, burnin = 10, thin = 1 | 2190.6 |
*Table 2:  Comparison in MCMC runtimes*

Note that, indeed, the hardware used in the reproduction process is not as powerful as what was used in the original study. Nevertheless, given the magnitude of the discrepancy, it is very unlikely that this was a main cause for it.

In addition, when this issue was first noticed, the possibility that the added runtime ocurred due to misuse of the authors’ code was, naturally, considered. To reduce the risk of such an error being responsible for the irreproducibility, the exisiting codebase was once more analyzed in order to find a example of how to construct, fit, and conduct inference with an MCMC model. Such an example was found in a method responsible for building from scratch and evaluating MCMC models and used as a comparison to correct possible mistakes. However, the re-implementation and the example implementation utilized the pipeline in the same manner. This agreement between the re-implementation and the authors’ own implementation make it very unlikely that an implementation error during reproduction is responsible for the issue at hand.
## Porting to pytorch:

|   | LCPFN | Large Model (Forced teaching) | Small Model (Forced teaching) | Small Model | Small  Model (Positional encoding: Euclidian distance)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Mean  |  **0.0002**  | 2.3589 | 3.2686 | 1.9210 | 3.4749 |
| Standard Deviation  | **8.0023e-05**  | 0.3185 | 0.5678 | 0.2005 | 0.5448 |

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



- Result and differences 
- why?



## Testing their model on new data
## Limitations
- Each makes a paragraph of our limitations
## Conculsion and future work

## References
[1] Adriaensen, S., Rakotoarison, H., Müller, S., & Hutter, F. (2023). *Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks*. arXiv preprint arXiv:2310.20447. Available at: https://arxiv.org/abs/2310.20447
[2] Domhan, T., Springenberg, J.T., & Hutter, F. (2015). *Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves*. In Proceedings of the Twenty-Fourth International Joint Conference on Artificial Intelligence (IJCAI 2015).