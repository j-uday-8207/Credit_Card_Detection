# Credit Card Fraud Detection

This project implements a credit card fraud detection system using a Hidden Markov Model (HMM) and various clustering algorithms. The system is designed to identify fraudulent transactions by analyzing spending patterns.

## Table of Contents

- [Project Description](#project-description)
- [File Descriptions](#file-descriptions)
- [How It Works](#how-it-works)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Project Description

This project utilizes a Hidden Markov Model (HMM) to detect credit card fraud. The process begins by clustering transaction amounts into several groups using algorithms like K-Means. These clusters are then used as observations for the HMM. The HMM is trained on a sequence of normal transactions for each credit card to learn the typical spending behavior. When a new transaction occurs, the system evaluates the probability of this new sequence of transactions. If the probability is significantly lower than a predefined threshold, the transaction is flagged as potentially fraudulent.

## File Descriptions

* `hidden_markov_main.py`: This is the main script that drives the fraud detection process. It reads transaction data, trains the HMM for each credit card, and evaluates new transactions for fraud.
* `hidden_markov_model.py`: This file contains the implementation of the Hidden Markov Model, including the forward and backward algorithms, and the Baum-Welch algorithm for training.
* `clustering.py`: This script provides various clustering algorithms, including K-Means, DBSCAN, and Agglomerative Clustering (Single, Complete, and Average Linkage), to group transaction amounts.
* `decisionTree.py`: Implements a Decision Tree classifier from scratch.
* `randomForestt.py`: Implements a Random Forest classifier using the custom `DecisionTree` class.
* `config.py`: A configuration file that stores constants and parameters used throughout the project, such as the fraud detection threshold, number of states in the HMM, and number of clusters.

## How It Works

1.  **Data Preparation**: The system reads credit card transaction data and groups it by credit card number.
2.  **Clustering**: For each card, the transaction amounts are clustered into a predefined number of groups using K-Means clustering. Each cluster represents a different spending level.
3.  **HMM Training**: A Hidden Markov Model is trained for each credit card using the sequence of transaction clusters. This training process allows the HMM to learn the normal transaction patterns for that specific card.
4.  **Fraud Detection**: To check a new transaction, it's first assigned to a cluster. Then, a new observation sequence is created by appending this new transaction's cluster to the recent history of transactions. The HMM calculates the probability of this new sequence. If the probability drops below a certain threshold compared to the previous sequence's probability, the transaction is flagged as fraudulent.

## How to Run

1.  **Prerequisites**: Ensure you have Python and the required libraries installed.
2.  **Dataset**: You will need `fraudTrain.csv` and `fraudTest.csv` files in the same directory as the scripts.
3.  **Execution**: Run the `hidden_markov_main.py` script from your terminal:
    ```bash
    python hidden_markov_main.py
    ```
4.  **Configuration**: You can adjust the parameters in `config.py` to fine-tune the model's performance. The main script will iterate through different thresholds to find the best F1-score.

## Dependencies

* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
