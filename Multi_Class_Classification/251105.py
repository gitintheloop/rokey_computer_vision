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
def plot_confusion_matrix_with_analysis(y_true, y_pred, class_names): # y이고 , y 헷이겠지
    cm = confusion_matrix(y_true, y_pred) # sklearn에 있어.

    fig, axes = plt.subplots(1, 2, figsize = (15,5)) # 왼쪽;혼동행렬히트맵, 오른쪽:상위오류(barh)

    # confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0]) # axes[0]으로 하면 왼쪽 서브플롯에 그림
    axes[0].set_title("Confusion Matrix", fontsize=14, pad=10)
    axes[0].set_ylabel('actual class', fontsize=11)
    axes[0].set_xlabel('predicted class', fontsize=11)

    # Error Analysis 
    # 실제 클래스(행)기준으로 비율을 만들기 위해 정규화
    # i행의 합으로 나눠 실제 i에서 j로 간 비율(오류율)
    # 0 으로 나눔 방지
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    error_analysis = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i,j] > 0:
                error_analysis.append({
                    'True': class_names[i],
                    'Pred': class_names[j],
                    'Count': cm[i,j],
                    'Rate': cm_normalized[i,j] 
                }) 

    if len(error_analysis) > 0: # 오분류가 있다면
        error_analysis.sort(key=lambda x: x['Rate'], reverse=True)
        top_errors = error_analysis[:min(5, len(error_analysis))]
        error_labels = [f"{e['True']}→{e['Pred']}" for e in top_errors]
        error_rates = [e['Rate'] * 100 for e in top_errors]

        axes[1].barh(error_labels, error_rates, color='coral')
        axes[1].barh('error rate(%)',fontsize=11)
        axes[1].barh('main types of error', fontsize=14, pad=10)
        axes[1].barh(axis='x', alpha=0.3)
    else: # 오분류가 없다면
        axes[1].text(0.5, 0.5, 'Perfect Classification',
                     transform=axes[1].transAxes, ha='center', va='center',
                     fontsize=16)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return cm

# --------------------------------------------------------

# 5. Calculate Detailed Metrics

# ---------------------------------------------------------
def calculate_detailed_metrics(y_true, y_pred, y_proba, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics_avg = {
        'macro': precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[:3],
        'micro': precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)[:3],
        'weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[:3],
    }

    fig, axes = plt.subplots(2, 2, figsize=(15,10))

    # (1) Per-class metrics
    x = np.arange(len(class_names))
    width = 0.25

    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('class', fontsize=11)
    axes[0, 0].set_ylabel('score', fontsize=11)
    axes[0, 0].set_title('calculate detailed metrics', fontsize=13, pad=10)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])

    # (2) Comparison of averaging methods (micro, macro, weighted)
    avg_types = ['Macro', 'Micro', 'Weighted']
    avg_precisions = [metrics_avg['macro'][0], metrics_avg['micro'][0], metrics_avg['weighted'][0]]
    avg_recalls = [metrics_avg['macro'][1], metrics_avg['micro'][1], metrics_avg['weighted'][1]]
    avg_f1s = [metrics_avg['macro'][2], metrics_avg['micro'][2], metrics_avg['weighted'][2]]

    x_avg = np.arange(len(avg_types))
    axes[0, 1].bar(x_avg - width, avg_precisions, width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x_avg, avg_recalls, width, label='Recall', alpha=0.8)
    axes[0, 1].bar(x_avg + width, avg_f1s, width, label='F1-Score', alpha=0.8)
    axes[0, 1].set_xlabel('average method', fontsize=11)
    axes[0, 1].set_ylabel('score', fontsize=11)
    axes[0, 1].set_title('comparison of averaging methods', fontsize=13, pad=10)
    axes[0, 1].set_xticks(x_avg)
    axes[0, 1].set_xticklabels(avg_types)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])
    
    # (3) Support vs F1
    axes[1, 0].scatter(support, f1, s=100, alpha=0.6, c=range(len(class_names)), cmap='viridis')
    for i, name in enumerate(class_names):
        axes[1,0].annotate(name, (support[i], f1[i]),
                           xytext=(5,5), textcoords='offset points', fontsize=9)
    axes[1, 0].set_xlabel('number of samples', fontsize=11)
    axes[1, 0].set_ylabel('F1-Score', fontsize=11)
    axes[1, 0].set_title('Class Imbalance and performance', fontsize=13, pad=10)
    axes[1, 0].grid(alpha=0.3)

    # (4) Explaination
    explanation = (
        ' average method: \n\n'
        " Macro: average per class \n"
        " every class \n\n"
        " Micro : based on all samples \n"
        " the influence of the majority class \n\n"
        " Weighted : weighted average \n"
        " Reflecting the actual distribution "
    )
    axes[1, 1].text(0.1, 0.5, explanation, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=3, zero_division=0))
    
    return metrics_avg