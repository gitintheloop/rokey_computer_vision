# 251105.py

# 10th Session : Advanced Evaluation Metrics and Class Imbalance Handling
# Confusion Matrix Interpretation 
# Precision, Recall, and F1-Score (Per-Class/Micro/Macro)
# Calibration and Temperature Scaling
# Class Imbalance Handling

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import(
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from collections import Counter

# Fixed Random Seed Reproducibility 
# Create same data every run
torch.manual_seed(42)
np.random.seed(42)

#-------------------------------------------------------

# 1. Imbalance Dataset

#----------------------------------------------------------
class ImbalanceDataset(Dataset):
    def __init__(self, n_samples=2000, n_features=20, n_classes=4,
                 imbalance_ratio=[0.5,0.3,0.15,0.05]):
        self.n_classes = n_classes
        samples_per_class = [int(n_samples * ratio) for ratio in imbalance_ratio] # 1000,600,300,100

        X_list = []
        y_list = []

        for class_idx in range(n_classes):
            n = samples_per_class[class_idx]
            mean = np.random.randn(n_features) * (class_idx + 1)
            cov = np.eye(n_features) * (0.5 + class_idx * 0.2)  # covariance # The variance increases as the class index gets larger.
            X_class = np.random.multivariate_normal(mean, cov, n)  # 다변량 정규분포 X생성 
            y_class = np.full(n, class_idx)                        # 해당 클래스번호로 채운 레이블 벡터 
            # Create data where each class has a different distribution    
            X_list.append(X_class)
            y_list.append(y_class)
        
        # X : all input data, Y : all labels, this is supervised learing. Convert to a Tensor.
        self.X = torch.FloatTensor(np.vstack(X_list)) # Use np.vstack to stack all class data vertically into a single array
        self.y = torch.LongTensor(np.vstack(y_list))  # Use np.vstack to stack all class data vertically into a single array  
        
        # Use Counter to check whether each class has the intended imbalance ratio
        class_counts = Counter(self.y.numpy())
        print(f"\nOverall class distribution : {dict(sorted(class_counts.items()))} ")

    def __len__(self):                      # Return the total number of samples in the dataset.
        return len(self.X)

    def __getitem__(self, idx):             # 쌍으로 반환한다. (feature, label)
        return self.X[idx], self.y[idx]

#--------------------------------------------------------------
    
 # 2. Model Definition

#---------------------------------------------------------------
class MulitClassClassifier(nn.Module): # FeddForward Neural Network MLP
    def __init__(self, input_dim=20, hidden_dim=64, n_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),

            nn.BatchNorm1d(hidden_dim),     # stabilize training, improve convergence
            nn.ReLU(),                      # non-linear activation function
            nn.Dropout(0.3),                # prevent overfitting 

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, n_classes) # n_classes : multiclass classification
        )                                    # logit(로짓) 은 “확률로 가기 전의 값”, logit -> softmax -> probability
                                             # 즉 Softmax나 Sigmoid를 통과하기 전의 원본 출력값. 
                                             # Logit is the raw, unnormalized output of a neural network before applying Softmax or Sigmoid.

    def forward(self, x):
        return self.network(x)
    
#------------------------------------------------------------

# 3. Temperature Scaling

#-------------------------------------------------------------
class TemperatureScaling(nn.Module): # It can controll sharpness of probalilty
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # initial value = 1.5

    def forward(self, logits):
        return logits / self.temperature # logit/T High T High Soft; Low T High Sharp
    
    def calibrate(self, model, val_loader, device, max_iter=50):
        nll_criterion = nn.CrossEntropyLoss() # CrossEntropyLoss가 최소가 되도록
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        # LBFGS is very effective for optimizing scalar parameter.
        logits_list = []
        labels_list = []

        model.eval() # 평가 모드로 전환 (Dropout, BN 정지)
        with torch.no_grad(): # 그래디언트 계산 비활성화 (속도↑, 메모리↓)
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        def eval_loss(): # 여러 번 loss를 평가하고 업데이트합니다.
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels) # logit/T
            loss.backward() # NLL이 작아지도록 T를 조정한다.
            return loss
        
        optimizer.step(eval_loss)

        print(f"Temperature Scaling Completed : T = {self.temperature.item():.3f}")
        return self.temperature.item()

# --------------------------------------------------------

# 4. Confusion Matrix Visualization

# ---------------------------------------------------------
def plot_confusion_matrix_with_analysis(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize = (15,5))

    # confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontsize=14, pad=10)
    axes[0].set_ylabel('actual class', fontsize=11)
    axes[0].set_title('predicted class', fontsize=11)
