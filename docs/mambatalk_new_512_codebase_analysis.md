# MambaTalk System Architecture

## 1. Motion Representation

Each frame of holistic human gesture is encoded as a 320-dimensional MHR (Mamba Human Representation) vector, decomposed into three semantic branches:

| Branch | Dim | Content |
|--------|-----|---------|
| Body | 130 | Upper/lower body joint rotations |
| Hands | 108 | Bimanual finger articulations (PCA 6D) |
| Global | 7 | Root orientation (3) + foot contact (4) |

Face parameters (expression 72d + jaw 3d = 75d) are included in the pose vector for autoregressive conditioning but are not actively predicted through VQ classification.

## 2. Pre-trained Motion Tokenizers (VQ-VAE)

Body and Hands branches are each independently tokenized by a pre-trained VQ-VAE, which remains frozen throughout the main model training. Each VQ-VAE consists of:

- **Encoder**: A stack of 1D convolutional layers with residual blocks, mapping raw motion parameters to a continuous latent space of dimension 256.
- **Quantizer**: A learned codebook of 256 entries. Each latent frame is discretized to its nearest codebook vector via L2 distance, with gradients propagated through the straight-through estimator.
- **Decoder**: A symmetric Conv1d stack that reconstructs the original motion parameters from quantized latent vectors.

All convolutions use kernel size 3 with stride 1, preserving temporal resolution. This design ensures that when the decoder processes a concatenated sequence of VQ indices from multiple clips, its receptive field naturally spans clip boundaries, providing implicit temporal smoothing.

Global motion (7d) is predicted as a continuous regression target without VQ tokenization.

## 3. Conditional Feature Encoding

The model is conditioned on three input modalities:

- **Audio**: Raw 16 kHz waveform processed by a 1D convolutional encoder, yielding per-frame features of dimension 512.
- **Text**: Time-aligned word tokens embedded via pre-trained FastText and linearly projected to 512 dimensions.
- **Speaker Identity**: A learnable embedding of dimension 768.

Audio and text features are fused via a learned softmax attention gate that computes a per-dimension weighted combination of the two modalities.

## 4. MambaTalk Network

The core generator follows a **Global-to-Local cascade** architecture:

### 4.1 Global Context Module

This module captures long-range temporal dependencies from the historical motion sequence. The input motion is first encoded by a multi-layer 1D convolutional encoder, then refined through a single-layer Transformer self-attention block to establish inter-frame structural relationships. A dual-path Mamba scan further models the sequential dynamics of both audio-driven and motion-driven features. The output provides a holistic motion context representation for the subsequent local generation stage.

### 4.2 Local Generation Module

This module performs fine-grained motion synthesis conditioned on the global context and audio-text features. An 8-layer Transformer decoder cross-attends from the global motion context (query) to the fused audio-text condition (memory), injecting speech-driven semantics into the motion representation.

The cross-attended features are then projected into three independent branches — body, hands, and global — each processed by a dedicated Mamba state space model for temporal refinement. The body and hands branches output 256-dimensional logits over the VQ codebook, while the global branch directly regresses 7-dimensional continuous parameters.

### 4.3 Output Specification

| Branch | Output | Decoding |
|--------|--------|----------|
| Body | 256-d codebook logits | argmax index → VQ-VAE decode → 130d |
| Hands | 256-d codebook logits | argmax index → VQ-VAE decode → 108d |
| Global | 7-d continuous values | Direct regression (root rotation + contact) |

## 5. Training Objectives

The model is optimized with a composite loss combining discrete classification, continuous reconstruction, and kinematic regularization:

**Codebook Classification Loss.** For body and hands, the predicted logits are supervised against ground-truth VQ indices. Body uses label-smoothed cross-entropy to mitigate codebook oscillation; hands uses standard negative log-likelihood.

**Latent Reconstruction Loss.** Mean squared error between the predicted latent vectors and the ground-truth quantized codebook entries (for body and hands) or raw parameters (for global), ensuring continuous-space alignment in addition to discrete classification.

**Velocity and Acceleration Loss.** First-order (velocity) and second-order (acceleration) temporal differences of the predicted latents are penalized via L1 loss against ground-truth differences, encouraging temporally smooth predictions.

**Self-supervised Masking.** During training, two additional forward passes are performed with progressively increasing random masking (ratio grows from 5% to 100% over training) applied to the input motion. One pass disables text conditioning while the other enables it, yielding auxiliary latent and classification losses that improve the model's robustness to partial observations and its capacity for autonomous generation.

## 6. Autoregressive Inference

At inference time, long sequences are generated through an autoregressive sliding-window scheme:

1. The sequence is divided into overlapping clips of 64 frames with 4-frame overlap.
2. For each clip, the model predicts VQ indices for body and hands, and continuous values for global motion.
3. The decoded output of each clip's final frames serves as the conditioning context for the next clip's initial frames.
4. After all clips are processed, the VQ indices from all clips are concatenated and decoded in a single pass through the VQ-VAE decoder, allowing the Conv1d receptive field to span clip boundaries.

The final 320-dimensional holistic gesture sequence is assembled by concatenating the decoded body (130d), hands (108d), face (75d, from autoregressive feedback), and global (7d) branches.
