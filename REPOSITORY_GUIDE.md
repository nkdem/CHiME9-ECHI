# CHiME9-ECHI Repository Study Guide

**For Beginners with Limited ML Experience**

This guide provides a detailed explanation of the CHiME9-ECHI repository structure, neural network architecture, and implementation details to help you understand, study, and improve the baseline system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Understanding the Neural Network Architecture](#3-understanding-the-neural-network-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Training Pipeline](#5-training-pipeline)
6. [Enhancement Pipeline](#6-enhancement-pipeline)
7. [Evaluation Pipeline](#7-evaluation-pipeline)
8. [Key Concepts for ML Beginners](#8-key-concepts-for-ml-beginners)
9. [How to Run and Test](#9-how-to-run-and-test)
10. [Potential Improvements](#10-potential-improvements)

---

## 1. Project Overview

### The Problem

People with hearing impairment struggle to understand conversations in noisy environments (like cafeterias). This project aims to enhance speech from a target speaker while suppressing background noise and competing speakers.

### The Solution

A neural network that takes:
- **Multi-channel audio** from hearing aids (4 channels) or Aria glasses (7 channels)
- **Speaker ID audio** (a sample of the target speaker's voice, called "rainbow passage")

And outputs:
- **Enhanced single-channel audio** of just the target speaker

### Why This Is Hard

1. **Real-world noise**: Cafeteria noise, background conversations
2. **Overlapping speech**: Multiple people talking simultaneously
3. **Low latency requirement**: Must work in near real-time for hearing aids
4. **Moving microphones**: People move their heads while talking

---

## 2. Repository Structure

```
CHiME9-ECHI/
├── config/                    # Configuration files (Hydra framework)
│   ├── train/                 # Training configurations
│   │   ├── main_ha.yaml       # Hearing aid training config
│   │   ├── main_aria.yaml     # Aria glasses training config
│   │   ├── model.yaml         # Model architecture parameters
│   │   ├── train.yaml         # Training hyperparameters
│   │   ├── dataloading.yaml   # Data loading settings
│   │   └── wandb.yaml         # Weights & Biases logging
│   ├── enhancement/           # Enhancement configurations
│   │   ├── main.yaml          # Main enhancement config
│   │   └── enhance_args/      # Model-specific enhancement parameters
│   │       ├── baseline.yaml  # Baseline model settings
│   │       └── passthrough.yaml
│   ├── evaluation/            # Evaluation configurations
│   │   ├── main.yaml
│   │   ├── metrics.yaml       # Full metrics suite
│   │   └── metrics_quick.yaml # Fast CPU metrics only
│   ├── paths.yaml             # All file paths
│   └── shared.yaml            # Shared parameters
│
├── src/                       # Source code
│   ├── shared/                # Shared utilities
│   │   ├── CausalMCxTFGridNet.py  # THE MAIN NEURAL NETWORK MODEL
│   │   ├── core_utils.py      # Model loading, device detection
│   │   └── signal_utils.py    # Audio processing utilities
│   ├── train/                 # Training modules
│   │   ├── echi.py            # Dataset class
│   │   ├── gromit.py          # Training tracker/logger
│   │   ├── losses.py          # Loss functions
│   │   └── signal_prep.py     # Signal preprocessing
│   ├── enhancement/           # Enhancement modules
│   │   ├── baseline.py        # Baseline enhancement implementation
│   │   ├── passthrough.py     # Simple passthrough
│   │   └── registry.py        # Plugin registry system
│   └── evaluation/            # Evaluation modules
│       └── segment_signals.py # Signal segmentation utilities
│
├── scripts/                   # Executable scripts
│   ├── train/                 # Training scripts
│   │   ├── train_script.py    # Main training loop
│   │   └── unpack.py          # Data unpacking/preparation
│   ├── enhancement/           # Enhancement scripts
│   │   ├── enhance.py         # Main enhancement script
│   │   └── resample.py        # Audio resampling
│   └── evaluation/            # Evaluation scripts
│       ├── setup.py           # Prepare reference signals
│       ├── validate.py        # Validate submission format
│       ├── prepare.py         # Prepare signals for evaluation
│       ├── evaluate.py        # Run metrics
│       └── report.py          # Generate reports
│
├── checkpoints/               # Pre-trained model weights
│   ├── ha_baseline.pt         # Hearing aid model
│   ├── aria_baseline.pt       # Aria glasses model
│   ├── ha_config.yaml         # HA model configuration
│   └── aria_config.yaml       # Aria model configuration
│
├── data/                      # Dataset (downloaded separately)
│   └── chime9_echi/           # Main dataset folder
│
├── enhancement_plugins/       # Custom enhancement plugins
│
├── run_train.py               # MAIN ENTRY: Train a model
├── run_enhancement.py         # MAIN ENTRY: Enhance audio
├── run_evaluation.py          # MAIN ENTRY: Evaluate results
└── run_evaluation_parallel.sh # Parallel evaluation script
```

---

## 3. Understanding the Neural Network Architecture

### Overview: MCxTFGridNet (Multi-Channel Time-Frequency GridNet)

**Location:** `src/shared/CausalMCxTFGridNet.py`

This is a **target speaker extraction** network that operates in the **time-frequency domain** (after STFT transformation).

### Why Time-Frequency Domain?

**For ML Beginners:**
- Audio is a 1D time signal (amplitude vs. time)
- STFT (Short-Time Fourier Transform) converts it to 2D (time × frequency)
- Think of it like a spectrogram: horizontal axis = time, vertical axis = frequency
- Neural networks work better on spectrograms for speech because:
  - Different sounds occupy different frequency bands
  - Noise and speech are easier to separate in frequency domain
  - Convolutions can capture patterns in both time and frequency

### Architecture Components

#### 1. **Input Embedding** (Lines 76-79)

```python
self.conv = nn.Sequential(
    nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
    LayerNormalization(emb_dim, eps=eps),
)
```

**What it does:**
- Takes STFT spectrogram (complex values = 2 channels per microphone)
- Hearing aids: 4 mics × 2 = 8 input channels
- Converts to `emb_dim` (64) feature channels
- **Purpose:** Compress raw STFT into a learned feature representation

**Why useful for this challenge:**
- Learns optimal features from multiple microphones
- Reduces dimensionality while preserving important information

#### 2. **Auxiliary Encoder** (Lines 81, 414-449: `AuxEncoder`)

```python
self.aux_enc = AuxEncoder(emb_dim, n_srcs)
```

**What it does:**
- Processes the "rainbow passage" (target speaker sample)
- Extracts a speaker embedding vector
- Uses a U-Net style encoder with downsampling

**Components:**
- **EnUnetModule**: Encoder blocks that progressively downsample
- **Output**: A fixed-size vector representing the target speaker's voice characteristics

**Why useful for this challenge:**
- Tells the network "this is the speaker we want to enhance"
- Helps distinguish target speaker from competing speakers
- Like showing the network a photo: "find this person in a crowd"

**ML Concept - Speaker Embeddings:**
- A compact numerical representation of someone's voice
- Similar voices have similar embeddings
- Used to condition the separation network

#### 3. **FiLM Layers** (Lines 83-85, 451-465: `FiLM`)

```python
self.fusions = nn.ModuleList([])
for _ in range(n_layers):
    self.fusions.append(FiLM(emb_dim, emb_dim))
```

**FiLM = Feature-wise Linear Modulation**

**What it does:**
```python
gamma = self.gamma_fc(cond)  # Scaling factor
beta = self.beta_fc(cond)     # Shifting factor
return gamma * x + beta        # Modulate features
```

**Why useful for this challenge:**
- Injects speaker identity into each processing layer
- Adapts the network's behavior based on who we're trying to extract
- Like a "tuning dial" that adjusts processing for each speaker

**ML Concept - Conditioning:**
- FiLM is a way to condition a network on external information
- Here: "modify your processing based on this speaker embedding"
- Common in conditional generation (e.g., "generate a red car")

#### 4. **GridNet Blocks** (Lines 87-100, 158-354: `GridNetV3Block`)

```python
self.gridnets = nn.ModuleList([])
for _ in range(n_layers):
    self.gridnets.append(GridNetV3Block(...))
```

**This is the core processing unit. Each block has:**

##### a) **Intra-RNN** (Lines 262-283)
- Processes **across frequency** at each time step
- LSTM (Long Short-Term Memory) processes frequency bins
- **Why:** Captures relationships between different frequencies
- **Example:** Harmonics of a voice span multiple frequencies

**ML Concept - LSTM:**
- A type of RNN (Recurrent Neural Network)
- Good at processing sequences (time or frequency)
- "Remembers" context from earlier in the sequence
- Here: bidirectional (looks both forward and backward)

##### b) **Inter-RNN** (Lines 287-309)
- Processes **across time** at each frequency bin
- LSTM processes time frames
- **Why:** Captures temporal evolution of speech
- **Example:** How a vowel sound evolves over time
- **Important:** Causal (unidirectional) for low-latency processing

**Why Causal Matters:**
- Can only look at past, not future frames
- Required for real-time hearing aid applications
- Adds ~10ms latency vs. 100ms+ for non-causal

##### c) **Self-Attention** (Lines 316-351)
- After RNNs, apply attention mechanism
- **Causal masking** (line 336-339): Only attend to past frames

**ML Concept - Self-Attention:**
- Allows the network to focus on important parts of the input
- Computes relationships between all time-frequency points
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I provide?"
- Attention weight: How much Q and K match

**Why useful for this challenge:**
- Speech has long-range dependencies (words span multiple frames)
- Can connect related speech components across time/frequency
- Multi-head attention (4 heads) captures different types of relationships

**ML Concept - Multi-Head Attention:**
- Multiple attention mechanisms in parallel
- Each "head" learns different patterns
- Head 1 might focus on pitch, Head 2 on energy, etc.

#### 5. **Output Decoder** (Line 102)

```python
self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
```

**What it does:**
- Converts learned features back to STFT domain
- Outputs complex mask (2 channels: real and imaginary)
- Applied to input spectrogram to extract target speaker

**Why useful:**
- Produces a "mask" that highlights target speaker frequencies
- Multiplication in frequency domain = selective filtering

### Overall Architecture Flow

```
Multi-channel STFT → Conv Embedding → [
    Speaker Embedding (from rainbow) →
    FiLM Layer (condition on speaker) →
    GridNet Block (Intra-RNN → Inter-RNN → Attention) →
    ... repeat N times (3 blocks) ...
] → Output Decoder → Target Speaker Mask → Apply Mask → Enhanced Speech
```

### Key Design Choices

1. **Stacked GridNet Blocks (3 layers):**
   - Gradual refinement of separation
   - Early layers: rough separation
   - Later layers: fine-grained enhancement

2. **LSTM Hidden Units (128):**
   - Balance between capacity and speed
   - More units = better quality but slower

3. **Attention Heads (4):**
   - Multiple perspectives on the data
   - More heads = more computational cost

4. **Small STFT Window (n_fft=128, ~8ms at 16kHz):**
   - Low latency (critical for hearing aids)
   - Trade-off: Less frequency resolution

### Why This Architecture Works for ECHI

1. **Multi-channel processing:**
   - Exploits spatial information from microphone array
   - Beamforming-like behavior learned implicitly

2. **Speaker conditioning:**
   - Distinguishes target from interferers
   - Generalizes to unseen speakers

3. **Causal design:**
   - Real-time capable
   - Only ~64ms algorithmic latency

4. **Time-frequency processing:**
   - Natural domain for speech enhancement
   - Mask-based separation is well-understood

---

## 4. Data Pipeline

### Dataset Structure

**Location:** `data/chime9_echi/`

```
chime9_echi/
├── dev/                       # Development set (10 sessions)
│   └── sessions/
│       └── [session_id]/
│           ├── aria/          # 7-channel Aria glasses audio
│           ├── ha/            # 4-channel hearing aid audio
│           ├── participants/  # Individual speaker recordings
│           └── segments/      # Speech segment timestamps
└── train/                     # Training set (39 sessions)
    └── sessions/
        └── [session_id]/
            ├── ...
```

### ECHI Dataset Class

**Location:** `src/train/echi.py`

**What it does:**
1. Loads CSV metadata of sessions and participants
2. Reads speech segment timestamps
3. Loads corresponding audio chunks during training

**Key Features:**
- **Segments:** Only loads 4-second speech segments (not full 36-minute sessions)
- **Three audio types per sample:**
  - `noisy`: Multi-channel device audio (HA or Aria)
  - `target`: Clean reference speech (single speaker)
  - `spkid`: Rainbow passage for speaker ID
- **Collation:** Batches samples with variable lengths, pads to max length

**Data augmentation potential:**
- Currently minimal augmentation
- Could add: pitch shifting, time stretching, noise injection

---

## 5. Training Pipeline

### Entry Point: `run_train.py`

**Pipeline:**
1. **Unpack** (`scripts/train/unpack.py`)
   - Resamples audio to model sample rate (16kHz)
   - Segments long recordings into speech-only chunks
   - Saves preprocessed data

2. **Train** (`scripts/train/train_script.py`)
   - Loads preprocessed segments
   - Trains model with backpropagation
   - Validates on dev set
   - Saves checkpoints

### Training Script Details (`scripts/train/train_script.py`)

#### Key Components

**Lines 136-142: Dataset Loading**
```python
trainset, trainsaves = get_dataset("train", data_cfg, debug)
devset, devsaves = get_dataset("dev", data_cfg, debug)
model = get_model(model_cfg, None)
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
loss_fn = get_loss(train_cfg.loss.name, train_cfg.loss.kwargs)
```

**Lines 162-220: Training Loop**

```python
for epoch in range(train_cfg.epochs):
    for batch in trainset:
        # 1. Load data
        noisy = batch["noisy"]      # Multi-channel input
        targets = batch["target"]   # Clean reference
        spk_id = batch["spkid"]     # Speaker ID

        # 2. Preprocess
        noisy = prep_audio(noisy, ...)   # Resample, normalize
        noisy = stft(noisy)              # Convert to STFT

        # 3. Forward pass
        processed = model(noisy, spk_id, lengths)

        # 4. Convert back to time domain
        processed = stft.inverse(processed)

        # 5. Compute loss
        loss = loss_fn(processed, targets)

        # 6. Backpropagation
        loss.backward()
        optimizer.step()
```

**Lines 221-285: Validation Loop**

- Every N epochs, evaluate on dev set
- Compute validation loss and STOI (speech intelligibility)
- Save audio samples for monitoring
- Save model checkpoint

### Loss Functions (`src/train/losses.py`)

**Available losses:**

1. **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**
   - Default loss for training
   - Measures signal quality
   - Scale-invariant: robust to volume differences

2. **STFT Loss**
   - Compares spectrograms
   - Captures perceptual quality

3. **Multi-Resolution STFT Loss**
   - Multiple STFT window sizes
   - Better frequency coverage

**ML Concept - Loss Functions:**
- Quantifies "how wrong" the model's prediction is
- Gradient descent minimizes this error
- Different losses emphasize different aspects (loudness, clarity, etc.)

### Training Tracker (`src/train/gromit.py`)

**What it does:**
- Logs training/validation losses
- Saves model checkpoints
- Saves audio samples for listening
- Integrates with Weights & Biases for visualization

**Key files created:**
- `exp/train_ha/checkpoints/*.pt` - Model weights
- `exp/train_ha/train_log.json` - Loss history
- `exp/train_ha/train_samples/` - Audio examples

---

## 6. Enhancement Pipeline

### Entry Point: `run_enhancement.py`

**Pipeline:**
1. **Resample** (`scripts/enhancement/resample.py`)
   - Resamples full session audio to model sample rate
   - One-time step, reuses for all models at same sample rate

2. **Enhance** (`scripts/enhancement/enhance.py`)
   - Loads full 36-minute session
   - Processes with enhancement model
   - Saves enhanced audio

### Baseline Enhancement (`src/enhancement/baseline.py`)

**Class: `Baseline`**

**Initialization (Lines 14-51):**
```python
def __init__(self, inference_dir, config_path, ckpt_path, ...):
    # 1. Load model configuration
    self.model_cfg = OmegaConf.load(config_path)

    # 2. Create STFT wrapper
    self.stft = STFTWrapper(**self.model_cfg.input.stft)

    # 3. Load trained model
    self.model = get_model(self.model_cfg, ckpt_path)
    self.model.eval()  # Inference mode

    # 4. Set up windowing parameters
    self.window_samples = window_size * sample_rate  # 60 seconds default
    self.stride_samples = stride * sample_rate        # 56 seconds default
    # Overlap: 4 seconds with crossfading
```

**Processing (Lines 55-131: `process_session`):**

```python
def process_session(self, device_audio, spkid_audio, ...):
    # 1. Preprocess audio
    device_audio = prep_audio(...)  # Resample, normalize
    spkid_audio = prep_audio(...)

    # 2. Process speaker ID once
    spkid_input = self.stft(spkid_audio)

    # 3. Process long audio in chunks
    output = torch.zeros(duration)
    for start in range(0, duration, self.stride_samples):
        # Get 60-second window
        snippet = device_audio[start:end]

        # Convert to STFT
        snippet = self.stft(snippet)

        # Run model
        enhanced_snippet = self.model(snippet, spkid_input, ...)

        # Convert back to time
        enhanced_snippet = self.stft.inverse(enhanced_snippet)

        # Crossfade overlaps to avoid clicks
        if overlap:
            enhanced_snippet *= hann_window

        # Accumulate output
        output[start:end] += enhanced_snippet

    return output
```

**Key Design:**
- **Windowing:** Processes long audio in overlapping chunks
- **Crossfading:** Smooth transitions between chunks using Hann window
- **Memory efficient:** Doesn't load entire 36 minutes into GPU at once

### Plugin System (`src/enhancement/registry.py`)

**Purpose:** Extensible architecture for custom models

**Usage:**
```python
from enhancement.registry import register_enhancement, Enhancement

@register_enhancement("my_model")
class MyModel(Enhancement):
    def process_session(self, device_audio, spkid_audio, ...):
        # Your enhancement logic
        return enhanced_audio
```

**Benefits:**
- Add new models without modifying core code
- Consistent interface for all models
- Easy comparison of different approaches

---

## 7. Evaluation Pipeline

### Entry Point: `run_evaluation.py`

**Pipeline:**
1. **Setup** (`scripts/evaluation/setup.py`)
   - Prepares reference signals (clean speech)

2. **Validate** (`scripts/evaluation/validate.py`)
   - Checks submission format
   - Verifies file names, sample rates, lengths

3. **Prepare** (`scripts/evaluation/prepare.py`)
   - Segments enhanced audio into speech chunks
   - Creates both "individual" and "summed" versions

4. **Evaluate** (`scripts/evaluation/evaluate.py`)
   - Runs metrics on each segment
   - Uses Versa evaluation toolkit

5. **Report** (`scripts/evaluation/report.py`)
   - Aggregates results
   - Generates JSON and CSV reports

### Evaluation Metrics

**Reference-based (requires clean speech):**
- **STOI:** Speech intelligibility
- **PESQ:** Perceptual quality
- **SI-SDR:** Signal-to-distortion ratio
- **Spectral metrics:** Compare frequency content

**Reference-free (no clean speech needed):**
- **DNSMOS:** DNN-based quality predictor
- **NOMAD:** Non-matching reference metric

### Segment Types

1. **Individual:** Each participant separately
   - Evaluates single-speaker extraction

2. **Summed:** All 3 conversation participants combined
   - Evaluates whether all conversation partners are preserved
   - Important: We want target speaker + conversation partners, not everyone

### Results Structure

```
working_dir/experiments/<EXP_NAME>/evaluation/reports/
├── report.dev.individual.ha._._.json         # All HA sessions, individual
├── report.dev.individual.aria._._.json       # All Aria sessions, individual
├── report.dev.summed.ha._._.json             # All HA sessions, summed
├── report.dev.summed.aria._._.json           # All Aria sessions, summed
├── report.dev.individual.ha.<session>._.json # Per session
└── report.dev.individual.ha.<session>.<pid>.json  # Per participant
```

**Key metrics to watch:**
- **weighted_mean:** Primary metric (weighted by segment length)
- **mean/std:** Distribution statistics
- Lower is better for DNSMOS P.808 MOS, higher for STOI/PESQ

---

## 8. Key Concepts for ML Beginners

### 1. Forward Pass vs. Backward Pass

**Forward Pass:**
- Input → Network → Output
- Prediction phase
- Used in both training and inference

**Backward Pass (Backpropagation):**
- Compute gradient of loss w.r.t. weights
- Update weights to reduce loss
- Only during training

### 2. Training vs. Inference

**Training:**
- `model.train()` mode
- Gradients computed and saved
- Dropout and batch norm active
- Takes hours/days

**Inference (Deployment):**
- `model.eval()` mode
- No gradients (faster, less memory)
- Dropout disabled, batch norm uses running stats
- Takes milliseconds per sample

### 3. Batching

**Why batch?**
- Process multiple samples simultaneously
- Exploits GPU parallelism
- Faster than processing one-by-one

**Variable length handling:**
- Pad shorter samples to match longest in batch
- Track original lengths to ignore padding

### 4. Normalization

**Layer Normalization (used in this model):**
- Normalizes features within each sample
- Helps training stability
- Reduces internal covariate shift

**RMS Normalization (for audio):**
- Normalizes loudness
- Prevents very loud/quiet samples from dominating

### 5. Attention Mechanism

**Intuition:**
- "Pay attention to relevant parts"
- Weighted combination of inputs
- Weights learned from data

**Example:**
- When enhancing "hello", attend to past "h" and future "o"
- Ignore irrelevant background chatter

### 6. Causal vs. Non-Causal

**Causal (used here):**
- Only uses past information
- Required for real-time systems
- Lower quality than non-causal

**Non-Causal:**
- Can use future information
- Better quality but higher latency
- Fine for offline processing

### 7. Loss Landscape & Optimization

**Loss function:**
- Surface in weight space
- Gradient descent: roll downhill
- Learning rate: step size

**Adam optimizer:**
- Adaptive learning rates per parameter
- Momentum: don't get stuck in local minima
- Standard choice for deep learning

---

## 9. How to Run and Test

### Setup (First Time)

```bash
# 1. Install dependencies
conda activate echi_recipe  # Or create environment
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 2. Download data (see README)
# Requires HuggingFace access token

# 3. Verify installation
python -c "import torch; import torchaudio; print('OK')"
```

### Quick Test: Enhancement Only

```bash
# Use pre-trained model to enhance dev set
python run_enhancement.py device=ha dataset=dev

# Output: exp/main/enhancement/ha/dev/*.wav
```

### Train a Model

```bash
# Train baseline on hearing aid data
python run_train.py

# Train on Aria data
python run_train.py device=aria

# Quick debug run (small dataset)
python run_train.py debug=true
```

**Monitor training:**
- Check `exp/train_ha/train_log.json` for losses
- Listen to samples in `exp/train_ha/train_samples/`
- View curves on Weights & Biases (if configured)

### Evaluate Your Model

```bash
# Full evaluation (slow)
python run_evaluation.py shared.exp_name=my_experiment \
    paths.enhancement_output_dir=exp/main/enhancement/ha/dev

# Quick evaluation (1/50 of data, CPU metrics only)
python run_evaluation.py shared.exp_name=my_experiment \
    paths.enhancement_output_dir=exp/main/enhancement/ha/dev \
    evaluate.score_config=config/evaluation/metrics_quick.yaml \
    evaluate.n_batches=50

# Results in: working_dir/experiments/my_experiment/evaluation/reports/
```

### Parallel Evaluation (Fast)

```bash
# Local machine with 8 cores
./run_evaluation_parallel.sh my_experiment exp/main/enhancement/ha/dev \
    --launcher local 8

# SLURM cluster
./run_evaluation_parallel.sh my_experiment exp/main/enhancement/ha/dev \
    --launcher slurm 40
```

### Experiment Workflow

```bash
# 1. Train modified model
python run_train.py model.params.n_layers=4

# 2. Enhance dev set with new model
python run_enhancement.py enhance_args.args.ckpt_path=exp/train_ha/checkpoints/epoch_29.pt

# 3. Evaluate
python run_evaluation.py shared.exp_name=4layers ...

# 4. Compare with baseline
# Check report files in evaluation/reports/
```

---

## 10. Potential Improvements

### For Beginners

#### 1. **Data Augmentation**

**Where:** `src/train/echi.py` `__getitem__` method

**Ideas:**
- Add random noise to training samples
- Pitch shifting (simulate different speakers)
- Time stretching (vary speech rate)
- Random volume changes

**Why it helps:** Model becomes more robust to variations

**Implementation sketch:**
```python
def __getitem__(self, index):
    # ... load audio ...

    # Data augmentation
    if self.training:
        # Add noise
        noise_level = random.uniform(0, 0.1)
        noisy += torch.randn_like(noisy) * noise_level

        # Random gain
        gain = random.uniform(0.8, 1.2)
        noisy *= gain
        target *= gain

    return out
```

#### 2. **Hyperparameter Tuning**

**Where:** `config/train/model.yaml`, `config/train/train.yaml`

**Parameters to try:**
- `n_layers`: Try 4, 5, 6 (currently 3)
- `lstm_hidden_units`: Try 192, 256 (currently 128)
- `attn_n_head`: Try 8 (currently 4)
- `emb_dim`: Try 96, 128 (currently 64)
- Learning rate: Try 5e-4, 1e-3 (currently 1e-3)

**Method:** Grid search or random search

#### 3. **Loss Function Combinations**

**Where:** `src/train/losses.py`, `config/train/train.yaml`

**Current:** SI-SDR only

**Try:**
- SI-SDR + STFT Loss (0.5 * si_sdr + 0.5 * stft_loss)
- Multi-resolution STFT
- Add perceptual loss (PESQ as loss)

**Implementation:**
```python
def combined_loss(output, target):
    sisnr = si_sdr_loss(output, target)
    stft = stft_loss(output, target)
    return 0.7 * sisnr + 0.3 * stft
```

#### 4. **Better Speaker Embeddings**

**Where:** `src/shared/CausalMCxTFGridNet.py`, `AuxEncoder`

**Ideas:**
- Use pre-trained speaker verification model (e.g., ECAPA-TDNN)
- Freeze embeddings from existing model
- Fine-tune on this task

**Why it helps:** Better speaker discrimination

### For Intermediate Users

#### 5. **Architecture Modifications**

**Ideas:**
- Add residual connections between GridNet blocks
- Use Conformer blocks instead of LSTM (convolution + attention)
- Dual-path processing (separate time and frequency more explicitly)

**Where:** `src/shared/CausalMCxTFGridNet.py`

#### 6. **Multi-Stage Training**

**Stage 1:** Train on short segments (4 seconds)
**Stage 2:** Fine-tune on longer segments (10-20 seconds)
**Stage 3:** Fine-tune with full context

**Why it helps:** Captures both local and global structure

#### 7. **Ensemble Methods**

- Train multiple models with different architectures
- Average predictions at inference
- Or use a meta-learner to combine

**Implementation:**
```python
output = 0.5 * model1(input) + 0.3 * model2(input) + 0.2 * model3(input)
```

#### 8. **Curriculum Learning**

- Start with easy examples (high SNR, low overlap)
- Gradually add harder examples
- Model learns progressively

**Where:** Modify `src/train/echi.py` to sort samples by difficulty

### For Advanced Users

#### 9. **Neural Beamforming**

- Explicitly model microphone array geometry
- Learn beamforming weights
- Combine with separation network

#### 10. **Self-Supervised Pre-Training**

- Pre-train on unlabeled audio (YouTube, podcasts)
- Task: Predict masked audio frames
- Fine-tune on ECHI data

#### 11. **Multi-Task Learning**

- Joint training: speaker extraction + noise suppression + dereverberation
- Shared encoder, multiple decoder heads

#### 12. **Adversarial Training**

- Add discriminator network
- Generator tries to fool discriminator
- Can improve perceptual quality

**Where:** Add discriminator in `src/train/losses.py`

### Testing Your Improvements

1. **Quantitative:** Check evaluation metrics (STOI, PESQ, SI-SDR)
2. **Qualitative:** Listen to enhanced audio samples
3. **Ablation study:** Test each change individually
4. **Statistical significance:** Run multiple random seeds

### Debugging Tips

**Model not learning (loss stays high):**
- Check data loading (visualize spectrograms)
- Verify loss function (print gradients)
- Reduce learning rate
- Check for NaN values

**Model overfitting (train loss low, val loss high):**
- Add data augmentation
- Increase dropout
- Reduce model capacity
- Use more training data

**Poor audio quality despite good metrics:**
- Listen to samples at different training stages
- Check for clipping, artifacts
- Visualize spectrograms
- Adjust loss function weights

---

## Appendix: Key Files Quick Reference

| Task | File | Purpose |
|------|------|---------|
| **Train model** | `run_train.py` | Main entry point |
| | `config/train/main_ha.yaml` | Configuration |
| | `src/train/train_script.py` | Training loop |
| | `src/shared/CausalMCxTFGridNet.py` | Model architecture |
| **Enhance audio** | `run_enhancement.py` | Main entry point |
| | `src/enhancement/baseline.py` | Baseline enhancement |
| | `config/enhancement/enhance_args/baseline.yaml` | Parameters |
| **Evaluate** | `run_evaluation.py` | Main entry point |
| | `config/evaluation/metrics.yaml` | Metric definitions |
| **Add custom model** | `src/shared/your_model.py` | New model class |
| | `src/shared/core_utils.py` | Register in `get_model()` |
| | `config/train/your_model.yaml` | Model config |
| **Add enhancement plugin** | `enhancement_plugins/your_plugin.py` | Plugin implementation |
| | `config/enhancement/enhance_args/your_plugin.yaml` | Plugin config |

---

## Further Reading

### Papers (Referenced in Code)

1. **TF-GridNet:** Wang et al., "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation" (ICASSP 2023)
2. **X-TF-GridNet:** Hao et al., "X-TF-GridNet: A time–frequency domain target speaker extraction network" (Information Fusion 2024)

### Concepts to Study

- **STFT and spectrograms:** Foundation for audio ML
- **LSTMs and RNNs:** Sequence modeling
- **Attention mechanisms:** Transformer basics
- **Speaker verification:** ECAPA-TDNN, x-vectors
- **Speech enhancement:** Wiener filtering, spectral subtraction
- **Evaluation metrics:** PESQ, STOI, SI-SDR

### Tools

- **PyTorch tutorials:** pytorch.org/tutorials
- **Hydra configuration:** hydra.cc
- **Weights & Biases:** wandb.ai (experiment tracking)
- **Versa evaluation toolkit:** (used in evaluation)

---

**Good luck with your research! Start simple, measure everything, and iterate.**