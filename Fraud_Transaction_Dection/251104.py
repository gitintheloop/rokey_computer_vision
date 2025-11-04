#251104.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import torch
import torch.nn as nn
import torch.optim as optim

# Visualization Settings
# plt.rcParams['font size'] = 12
# plt.rcParams['figure.figsize'] = (10,6)
# plt.rcParams['axes.grid'] = True
             
# Data Generation (Class Imbalance)
# Normal Transaction : 95%
# Fraud Transaction : 5%

# Let's create imbalanced data (Normal : 95%, Fraud : 5%)

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95,0.05],
    flip_y=0.01,
    random_state=42
)

# Check class distribution
np.unique(y, return_counts=True)  # unique() removes duplicate values
unique, counts = np.unique(y, return_counts=True) # np.unique() shows the number of samples in each class
print("normal_transaction(0): ", counts[0])
print("fraud_transaction(1): ", counts[1])

# Data Split (Stratified Sampling)

X_train, X_test, y_train, y_test =\
train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

