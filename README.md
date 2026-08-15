# MahlerVAE

<p align="center">
  <img src="assets/architecture.png"
       alt="Architecture"
       width="850">
</p>

### Motivation

<p align="justify">
This project sprang from MIT 6.S191's Lab 1 Exercise, which was about MIDI music generation. Their model used an LSTM to generate monophonic Irish folk tunes.
Inspired, I wanted to take it a step further by building a more powerful GPT model for polyphonic generation of symphonies in the style of Gustav Mahler.
"There was a similar venture called MahlerNet, but their model used LSTMs"
Hence, I built and trained a GPT model (seq_length=512) in PyTorch using a small dataset consisting of 281 pieces by Late Romantic composers--specifically, pieces composed by Beethoven, 
Brahms, Bruckner, Dvorak, Holst, Mahler, Sibelius, Strauss, Tchaikovsky, and Wagner.
The GPT model was able to generate convincing continuations given a musical idea, but it could not achieve long-term coherence, frequently generated repetitive motifs, and 
most importantly, lacked musical direction. 
</p>

### Overview
The model uses the Optimus architecture presented in [this paper](https://arxiv.org/pdf/2004.04092). This time, however, we use it as a MIDI generation model.
The idea is that, instead of using only a transformer decoder, a transformer encoder first represents a sequence of tokens as a variable in a low-dimensional latent space.
Thus, during training, the model learns a latent embedding space of musical ideas. We then take a sample z from the learned latent variable through a reparameterization trick:

$$z = \mu + \epsilon \odot \exp\left(\frac{1}{2} \log(\sigma^2)\right)$$

where $\mu$ and $\epsilon$ are vectors derived from the latent variable. During decoding, z is projected into a MEM vector $$h_{MEM}$$: a key that can be used by the decoder's self-attention blocks.
Specifically, $$h_{MEM}$$ is an additional key vector (and hence a value vector) that queries can attend to.
> [!NOTE]
> The authors of the Optimus architecture proposed two different forms of latent injection: *Memory*, the one we use, and *Embedding*. *Memory* was found to be significantly more effective than *Embedding*.
