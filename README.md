# NLP – Sentiment Analysis

## Author

Hugo Degeneve

## Macro-average accuracy on the dev data split

The method achieves 87.31 % macro-average accuracy on the dev data split. This was the average accuracy computed over the default 5 runs with :

```bash
accelerate launch runproject.py
```

## Implementation

### Multi-device Training

The code was designed to support multi-GPU training by relying on **Accelerate** to handle device assignment automatically. However, I did not have access to multiple GPUs to fully test this feature.

### VRAM

This project uses **XLM-RoBERTa-base** with a batch size of 8 and gradient accumulation of 4. It runs on less than **10 GB of VRAM**.

---

## Data Analysis

### Class Imbalance

The dataset exhibits a strong class imbalance across all aspects (**Price**, **Food**, **Service**), with a clear dominance of the *Positive* and *No Opinion* classes.

#### Train Set

**Price**

* No Opinion: 2564
* Positive: 728
* Negative: 562
* Mixed: 154
* Positive#NE: 1

**Food**

* Positive: 2773
* Negative: 536
* No Opinion: 383
* Mixed: 317

**Service**

* Positive: 2362
* No Opinion: 807
* Negative: 651
* Mixed: 189

#### Validation Set

**Price**

* No Opinion: 387
* Positive: 116
* Negative: 80
* Mixed: 17

**Food**

* Positive: 420
* Negative: 75
* No Opinion: 59
* Mixed: 46

**Service**

* Positive: 340
* No Opinion: 120
* Negative: 114
* Mixed: 26

Overall, the class distributions are consistent between the training and validation sets, which is desirable for reliable evaluation.

A simple baseline consisting of always predicting the majority class yields a macro accuracy of **64.0%**.

---

### Review Length

The dataset contains relatively short reviews:

* Fewer than **0.5%** of samples exceed 512 tokens.

This means truncation has minimal impact, and most reviews can be processed without loss of information.

---

### Noisy Annotations

A significant challenge in this dataset is the presence of **label noise**. Some annotations are inconsistent with the actual content of the reviews.

#### Examples

* **Mismatch between text and label**

  > *"En direction de Gap nous avons fait une halte dans cet établissement que l'on nous a conseillés ! Accueil très pro, un cadre très sympa et original, des prix raisonnables..."*
  > → **Price** labeled as *No Opinion*, despite an explicit mention of "prix raisonnables".

* **Label without textual evidence**

  > *"Super resto. Déco très sympa. Personnel au top. Cuisine super bonne et plats copieux (surtout les fameuses salades). Merci vraiment à toute l’équipe pour cette soirée réussie. Je le recommande vivement."*
  > → **Price** labeled as *Positive*, although price is not mentioned.

* **Incorrect sentiment polarity**

  > *"2h pour un repas classique... hors de prix et serveurs complètement désorganisés... attrape touriste."*
  > → **Price** labeled as *Mixed* (should be *Negative*)
  > → **Food** labeled as *Negative* (not clearly supported)

These inconsistencies justify the use of **robust training strategies** to mitigate the impact of noisy labels.

---

### Review Quality

The raw text data also contains formatting issues:

* Repeated quotation marks (`""""`) at the beginning or end of reviews
* Presence of escape characters such as backslashes (`\`)
* Irregular spacing and special characters

---

## Method

The proposed method aims to learn robustly from noisy annotations. It is inspired by:

**SelfMix: Robust Learning Against Textual Label Noise with Self-Mixup Training (Dan Qiao et al.)**

This method achieves 87.31 % macro-average accuracy on the dev data split. This was the average accuracy computed over the default 5 runs with :

```bash
accelerate launch runproject.py
```

---

### Review Preprocessing

We apply a series of cleaning steps to normalize and standardize the text:

* **Unicode normalization** ensures consistent encoding across inputs
* **Emoji removal** eliminates non-informative symbols
* **Quote normalization** standardizes different quotation styles
* **Removal of backslashes and redundant quotation marks**
* **Filtering of control characters** while preserving line breaks
* **Removal of non-standard symbols**, keeping only letters, digits, and common punctuation
* **Whitespace normalization** to ensure consistent formatting

This preprocessing improves input consistency and reduces noise before feeding data into the model.

---

### Model Architecture

We use **XLM-RoBERTa-base** as the text encoder, a multilingual transformer well-suited for French reviews.

On top of the shared encoder, we add **three independent classification heads**, one for each aspect:

* Price
* Food
* Service

Each head is a small feedforward neural network with:

* One hidden layer
* ReLU activation

This design introduces minimal additional complexity while allowing aspect-specific predictions.

---

## Training Strategy

The training procedure is designed to reduce the impact of noisy labels and stabilize learning.

### Stage 1: Head Training (Frozen Encoder)

* The transformer backbone is **frozen**
* Review embeddings are precomputed
* Only the classification heads are trained

This prevents early corruption of pretrained representations and allows the model to identify difficult or inconsistent samples.

---

### Removing Likely Noisy Samples

Following *SelfMix*, noisy samples are expected to produce **higher training loss**.

**Procedure:**

1. Compute the loss for each training sample after Stage 1
2. Rank samples by loss
3. Keep only the top $\tau$ fraction of samples (lowest loss)

This acts as a simple yet effective **data cleaning mechanism**. We do this after phase 1 as the model only learned the general meaning of each categories. As the authors of *SelfMix* claim this is the right time to do the filter because the model couldn't have overfited those labels yet.

---

### Stage 2: Full Model Training

After filtering:

* The full model (encoder + heads) is trained end-to-end
* A **Mixup-based regularization** strategy is applied

---

### Regularization via Mixup

Inspired by Mixup in computer vision, we interpolate between training examples in the embedding space.

Given two samples $(x_i, y_i)$ and $(x_j, y_j)$ we create a new one $(\tilde{x} , \tilde{y})$:

$$
\tilde{x} = \lambda \times \text{encoder}(x_i) + (1 - \lambda) \times \text{encoder}(x_j)
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where:

* $\lambda \sim \text{Beta}(\alpha, \alpha)$
* $\tilde{x}$ represents the encoder embeddings that is then given to the heads
* $\tilde{y}$ represents the labels, it is a soft label.

This approach:

* Encourages smoother decision boundaries
* Reduces overfitting
* Improves robustness to label noise

We apply this regularization with a probability of `mix_prob`.

---

## Hyperparameter Tuning

Hyperparameters were tuned using a Bayesian search with reduced training epochs:

* Phase 1 training: 30 epochs
* Phase 2 training: 2 epochs (instead of 4)

### Search Space

```yaml
head_learning_rate:
  distribution: log_uniform_values
  min: 1e-5
  max: 1e-3

learning_rate:
  distribution: log_uniform_values
  min: 1e-6
  max: 1e-4

weight_decay:
  distribution: log_uniform_values
  min: 1e-4
  max: 1e-2

mix_alpha:
  distribution: uniform
  min: 0.2
  max: 0.8

mix_prob:
  distribution: uniform
  min: 0.2
  max: 0.8

tau:
  distribution: uniform
  min: 0.8
  max: 1
```


### Final Values

```python
num_epochs = 4 # number of epochs in phase 2
num_epochs_head = 30 # number of epochs in phase 1
train_batch_size = 8

head_learning_rate = 7.274375606071262e-05 # lr for phase 1
learning_rate = 1.1829176048272468e-05 # lr for phase 2
weight_decay = 6.595780469253143e-04

grad_acc = 4
mix_alpha = 0.5195329578465342 # alpha for the mixup in phase 2
mix_prob = 0.2147598062440297 # probability of applying mixup in phase 2
tau = 0.9740003558057064 # tau parameter used to clean noisy samples after phase 1
```

## SelfMix evaluation

The addition of the *selfmix* method improved the macro-accuracy by around 0.1%. It is not clear from this result that the method is significantly better. To evaluate it more carefully, a manually curated test dataset of 20 samples was selected, and the following results were obtained:

* 17/20 without *selfmix*
* 19/20 with *selfmix*

This suggests a larger improvement than what was observed on the validation dataset. However, with only 20 test samples, it remains unclear whether the method provides a meaningful improvement over the baseline.

---

## File Structure

* `model.py`: Contains the implementation of the three-head classification model, as well as the head-only wrapper. The SelfMix method is also implemented here.
* `data.py`: Contains the data loading pipeline.
* `utils.py`: Contains constants and normalization functions.