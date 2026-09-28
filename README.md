# BA 64061 — Assignment 1: Neural Networks

IMDB movie-review sentiment classification with Keras. Extends the two-layer
ReLU network from class and measures how one change at a time affects
performance: hidden-layer depth, layer width, loss function, activation
function, and regularization (dropout, L2).

## Files

- `Assignment1_Report.pdf` — summary report: objective, approach, results,
  findings, conclusions, recommendation.
- `Assignment1_IMDB.ipynb` — the full experiment notebook (code, outputs,
  plots).

## Method

Fixed protocol for every run, as in class:

- Inputs: one-hot (multi-hot) encoded reviews, 10,000 most frequent words
- Optimizer: rmsprop; 20 epochs; batch size 512
- Validation: first 10,000 training samples; rest for training
- Test: 25,000 held-out reviews

## Experiments

| # | Configuration | Task |
|---|---------------|------|
| 1 | Two 16-unit ReLU hidden layers (baseline) | Baseline |
| 2 | Three 16-unit hidden layers | Depth |
| 3 | Two 64-unit hidden layers | Width |
| 4 | MSE loss | Loss function |
| 5 | Tanh activation | Activation |
| 6 | Dropout 0.5 | Regularization |
| 7 | L2 regularization (0.001) | Regularization |

## How to run

Open `Assignment1_IMDB.ipynb` in Google Colab and run all cells in order.
Each model trains on the IMDB dataset (downloaded automatically) and prints
final validation accuracy, best validation accuracy, test loss, and test
accuracy.

## Key result

Dropout (0.5) gave the best test accuracy and the most stable validation
performance; L2 regularization was a close second. Depth and width changes
added no accuracy and raised test loss — classic overfitting.
