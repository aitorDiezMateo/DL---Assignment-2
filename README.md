# Movie Recommendation System with Neural Collaborative Filtering and Autoencoders

This project explores several deep learning techniques for building a **Recommendation System**, with a focus on **Neural Collaborative Filtering (NCF)** and **Autoencoder-based models**. The goal is to predict discrete user ratings (1 to 5) for movies, leveraging both collaborative filtering signals and side information (e.g., user demographics and movie genres).

---

## Motivation

Recommender systems are central to modern platforms such as Netflix and Amazon. This project was developed to:
- Investigate deep learning-based recommender architectures.
- Compare the performance of NCF models with different loss functions (classification vs. regression).
- Explore hybrid models that integrate collaborative and content-based strategies.
- Evaluate the potential of Autoencoder-based models in capturing complex, non-linear relationships.

---

## Models Implemented

### 1. Neural Collaborative Filtering (NCF) with NeuMF Architecture
- **Approach:** Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
- **GMF Branch:** Models linear interactions via the element-wise product of user and film embeddings.
- **MLP Branch:** Processes concatenated embeddings through fully connected layers with non-linear activations (e.g., ReLU) to capture higher-order interactions.
- **Variants:**
  - **Classification:** Uses Cross-Entropy Loss for a probabilistic interpretation of ratings.
  - **Regression:** Uses Mean Squared Error (MSE) Loss for direct estimation of rating values.
- **Regularization:** Dropout, Batch Normalization, and noise injection are used to combat overfitting.

### 2. Hybrid NCF + Content-Based Models
- **Approach:** Enhances the base NCF model by incorporating auxiliary features such as movie genres and user demographics (age, occupation).
- **Variants:**
  - **Classification:** Uses Cross-Entropy Loss.
  - **Regression:** Uses Mean Squared Error (MSE) Loss.
- **Objective:** To capture more comprehensive patterns in user preferences by enriching the feature space.

### 3. Autoencoder-Based Recommendation System
- **Rationale:** After researching different approaches, Autoencoders emerged as an interesting alternative despite their higher implementation complexity compared to NCF. This model was implemented to assess how well Autoencoders capture complex, non-linear relationships between users and items, and to evaluate their performance relative to NCF models.
- **Support:** The implementation was aided by Generative AI tools and educational resources from Vibe Coding, specifically using platforms such as Cursor AI and Claude Sonnet 3.7.
- **Architecture:**
  - **Input:** Concatenation of learned embeddings (for user and item IDs) with explicit user/item features.
  - **Encoder:** A series of fully connected layers with Batch Normalization, LeakyReLU activations, and Dropout to extract robust latent representations.
  - **Variational Option:** Allows the latent space to be parameterized by a mean and log variance, facilitating sampling via the reparameterization trick.
  - **Decoder:** Maps the latent representation back to a predicted rating, enhanced by a residual connection from the input.
  - **Strength:** Leverages the Autoencoder’s ability to uncover hidden structure in the data, providing a valuable complement to the other recommendation models.

---

## Dataset

The project uses the **MovieLens** dataset, which includes:
- User ratings for movies.
- Movie genres.
- User demographic data (e.g., age, occupation).

Preprocessing steps involved encoding categorical variables, generating embedding indices, and feature scaling.

---

## Evaluation Metrics

The models are evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **Accuracy** (for classification models)
- **Loss evolution** over training epochs

Regularization techniques and early stopping were used to address overfitting.

---