# NLP – Sentiment Analysis

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

Overall, the class distributions are consistent between the training and validation sets, which is desirable for evaluation reliability.

A simple baseline consisting of always predicting the majority class yields a macro accuracy of **64.0%**.

---

### Review Length

The dataset contains relatively short reviews:

* Fewer than **0.5%** of samples exceed 512 tokens.

This means truncation has minimal impact and most reviews can be processed without loss of information.

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

The proposed method aims to learn robustly from noisy annotations. It is inspired by the approach introduced in:
**"SelfMix: Robust Learning Against Textual Label Noise with Self-Mixup Training" (Dan Qiao et al.)**

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
* ReLU

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

Procedure:

1. Compute the loss for each training sample after Stage 1
2. Rank samples by loss
3. Remove the top **x% highest-loss samples**

This acts as a simple yet effective **data cleaning mechanism**.

---

### Stage 2: Full Model Training

After filtering:

* The full model (encoder + heads) is trained end-to-end
* A **Mixup-based regularization** strategy is applied

---

### Regularization via Mixup

Inspired by Mixup in computer vision, we interpolate between training examples in the embedding space.

Given two samples $(x_i, y_i)$ and $(x_j, y_j)$:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where:

* $\lambda \sim \text{Beta}(\alpha, \alpha)$
* $x$ represents the encoder embeddings
* $y$ represents the labels (one-hot encoded)

This approach:

* Encourages smoother decision boundaries
* Reduces overfitting
* Improves robustness to label noise
