# MahlerVAE

<p align="center">
  <img src="assets/architecture.png"
       alt="Architecture"
       width="850">
</p>

## Motivation

<p align="justify">
This project sprang from MIT 6.S191's Lab 1 Exercise, which was about MIDI music generation. Their model used an LSTM to generate monophonic Irish folk tunes.
Inspired, I wanted to take it a step further by building a more powerful GPT model for polyphonic generation of symphonies in the style of Gustav Mahler.
"There was a similar venture called MahlerNet, but their model used LSTMs"
Hence, I built and trained a GPT model (seq_length=512) in PyTorch using a small dataset consisting of 281 pieces by Late Romantic composers--specifically, pieces composed by Beethoven, 
Brahms, Bruckner, Dvorak, Holst, Mahler, Sibelius, Strauss, Tchaikovsky, and Wagner.
The GPT model was able to generate convincing continuations given a musical idea, but it could not achieve long-term coherence, frequently generated repetitive motifs, and 
most importantly, lacked musical direction. 
</p>

## Tokenization scheme
We use a [Compound Word tokenization scheme](https://ojs.aaai.org/index.php/AAAI/article/view/16091) for faster training. This means that each time step is represented by a tuple of categorical fields, instead of a single event token drawn from one vocabulary. In particular, we have the following eleven fields/token attributes:

     Token attributes by idx
     0: family (BOS, EOS, Metric, Note, or PAD)
     1: bar_position
     2: channel
     3: program
     4: pitch
     5: velocity
     6: duration
     7: tempo
     8: time_signature
     9: controller_type
     10: controller_value


## Architecture
The model uses the Optimus architecture presented in [this paper](https://arxiv.org/pdf/2004.04092). This time, however, we use it as a MIDI generation model.
The idea is that, instead of using only a transformer decoder, a transformer encoder first maps a sequence of tokens into parameters of a low-dimensional latent distribution $q(z|x)$. This distribution is usually intended to follow a Gaussian distribution, so the encoder outputs parameters $\mu$ and $\log(\sigma^2)$. Thus, during training, the model learns a continuous latent space intended to capture higher-level musical characteristics. We then take a sample $\mathcal{z}$ from the learned latent variable through a reparameterization trick:

$$z = \mu + \epsilon \odot \exp\left(\frac{1}{2} \log(\sigma^2)\right)$$

where we sample $\epsilon$ ~ $\mathcal{N}(0, \mathcal{I})$. During decoding, z is projected into a MEM vector $$h_{MEM}$$: a key that can be used by the decoder's self-attention blocks.
Specifically, $$h_{MEM}$$ is an additional key vector (and hence a value vector) that queries can attend to, as shown below.

> The authors of the Optimus architecture proposed two different forms of latent injection: *Memory*, the one we use, and *Embedding*. *Memory* was found to be significantly more effective than *Embedding*, although combining both gave slightly better performance.

<p align="center">
  <img
    src="assets/attention matrix.png"
    alt="Decoder Attention Matrix"
    width="700"
  ><br>
  <i>Visualization purposes only. This does not reflect values in the actual attention matrices in the model.</i>
</p>

The intended effect of this memory vector is to improve long-term coherence by providing the autoregressive decoder with persistent access to the latent representation $\mathcal{z}$ across all attention layers, 
rather than forcing all information to propagate solely through previously generated tokens, as is the case with decoder-only transformers.

## Training
There are two terms in the loss function that we are trying to optimize:

1. **Latent loss ($$L_{KL}$$)**: measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence. If D denotes latent dimension and B denotes batch size, its loss is given by:

$$\mathcal{L}_{\mathrm{KL}}=\frac{1}{2DB}\sum_{i=1}^{D}\sum_{j=1}^{B}\left(\sigma_{ij}^{2}+\mu_{ij}^{2}-1-\log\sigma_{ij}^{2}\right)$$
      
2. **Reconstruction loss ($$L_{x}$$)**: standard cross-entropy loss between logits and labels. Since we are using a compound word tokenization scheme, we need to compute a loss for each field. If $$W_{F}$$ denotes the field weights (set via config.yaml), the loss term is given by:

$$\mathcal{L_{x}}=\frac{\sum_F W_F \cdot \mathrm{CrossEntropy}(\mathrm{logits}_F,\mathrm{labels}_F)}{\sum_F W_F}$$
   
> In the code, we use a mask to dismiss losses associated with "IGNORE" and "PAD" labels as they are not meaningful information that we 
want the model to learn.

The total loss is given by:

$$L_{total} = L_{x} + \beta \cdot L_{KL}$$

where $\beta$ is the weight we assign to the KL loss. 

### Preventing posterior collapse
Perhaps the most common failure mode of Transformer-VAEs is posterior collapse--a phenomenon where the decoder is strong enough that it learns to completely ignore the latent variable produced by the encoder, in which case the model effectively collapses to a decoder-only transformer. It may still perform well, but it loses the benefit given by a VAE architecture, which is the learned latent representation. We have implemented three ways to avoid this:

1. **Make the decoder intentionally weaker**\
   Perhaps the most obvious solution, we intentionally weaken the decoder so that it learns to use the latent representation $\mathcal{z}$. In *config1.yaml*, which is the configuration we use to train our sample model,      we use the following hyperparameters for both encoder and decoder:
   
              # Model
              latent_dim: 64
              attribute_embedding_dim: 32
              block_size: 1024
              d_model: 256
              num_heads: 8
              dropout: 0.1
              num_layers: 4
   
2. **KL thresholding**\
   Normally, the $L_{KL}$ objective tries to drive each of its elements $KL_{i}$ towards zero in a step known as regularization. However, that pushes the i'th dimension's posterior to look nearly identical to the prior, carrying very little information about     $x$. With thresholding, we set some threshold *dim_target_kl* (via *config.yaml*) below which the objective no longer tries to optimize that dimension's $KL_{i}$. In the code, this is equivalent to manually setting        KL_{i} to zero once its value goes below dim_target_kl. Thus, KL thresholding gives each latent dimension some free capacity to encode information
3. **KL annealing**\
   KL annealing lets the model learn to use $z$ before heavily regularizing $z$. Instead of starting training with the full KL weight $\beta$, we define a *max_beta* that we gradually climb up to during the course of training. *ratio_zero* defines a threshold such that the current $\beta$ is zero if *percent_iteration* = *num_iterations*/*total_iterations* is below *ratio_zero*, after which the current $\beta$ linearly increases to max_beta until *percent_iteration* reaches *ratio_zero* + *ratio_increase*. The hyperparameters *max_beta*, *ratio_zero*, and *ratio_increase* must be defined in *config.yaml*.

### Results from training the sample model



