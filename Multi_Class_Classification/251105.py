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
        self.y = torch.LongTensor(np.hstack(y_list))  # Use np.vstack to stack all class data vertically into a single array  
        
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
            nn.Linear(input_dim, hidden_dim), # input layer

            nn.BatchNorm1d(hidden_dim),     # stabilize training, improve convergence
            nn.ReLU(),                      # non-linear activation function
            nn.Dropout(0.3),                # prevent overfitting 

            nn.Linear(hidden_dim, hidden_dim), # hidden layer
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
        # LBFGS(Limited Memory Broyden Fletcher Goldfarb Shanno Optimization)
        # 2차 미분 최적화 알고리즘 : 뉴턴 방법의 근차 형태 (근사치)
        # >> 이전 단계 변화량을 사용, 헤시안(Hesian) 근사치
        # >> 실제 수치해석에 들어가는 헤시안을 직접 구하지 않고 점진적으로 업데이트
        # >> 최근 10단계 업데이트된 정보만 저장해서 근사치를 구함 >> 제한된 메모리 사용
        # (일반적으로 딥러닝에 사용하는 Adam, SGD 1차 미분: gradient)
        # ROS에서도 많이 쓰임
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
        
        logits = torch.cat(logits_list) # 붙인다
        labels = torch.cat(labels_list)

        def eval_loss(): # 여러 번 loss를 평가하고 업데이트
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels) # logit/T
            loss.backward() # NLL이 작아지도록 T를 조정
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
        error_analysis.sort(key=lambda x: x['Rate'], reverse=True) # sort오름차순으로 정렬,,x['Rate']를 기준으로내림차순으로 정렬
        top_errors = error_analysis[:min(5, len(error_analysis))]
        error_labels = [f"{e['True']}→{e['Pred']}" for e in top_errors] # 진짜인데 에러로 표현이 됐다
        error_rates = [e['Rate'] * 100 for e in top_errors]

        axes[1].barh(error_labels, error_rates, color='coral') # 바그래프인데 수평그래프
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
def calculate_detailed_metrics(y_true, y_pred, y_proba, class_names): # proba : 퍼센티지
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
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
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

#064617

# --------------------------------------------------------

# 6. ROC - AUC

# ---------------------------------------------------------
def plot_roc_curbes_multiclass(y_true, y_proba, class_names):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ROC per class
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
        
    axes[0].plot([0,1], [0,1], 'k--', lw=1, label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positibe Rate', fontsize=11)
    axes[0].set_ylabel('True Positibe Rate', fontsize=11)
    axes[0].set_title('ROC per class (One-vs-Rest)', fontsize=13, pad=10)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Macro/Micro
    axes[1].plot(fpr["micro"], tpr["micro"],
                label=f'Micro-avg (AUC={roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':',linewidth=3)
    axes[1].plot(fpr["macro"], tpr["macro"],
                label=f"Macro-avg (AUC={roc_auc["macro"]:.3f})",
                color='navy', linestyle=':',linewidth=3)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('Average ROC', fontsize=13, pad=10)
    axes[1].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.show()

    print("\n" + '='*60)
    print("ROC-AUC score")
    print("="*60)
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {roc_auc[i]:.4f}")
    print(f"{'Micro-avg':15s}: {roc_auc['micro']:.4f}")
    print(f"{'Macro-avg':15s}: {roc_auc['macro']:.4f}")
          
    return roc_auc

# --------------------------------------------------------

# 7. Utility Functions (IndexError Fixed)

# ---------------------------------------------------------
def get_labels_from_loader(loader):
    # Extract all labels from Dataloader
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.numpy().tolist())
    return labels

def get_class_weights(train_loader, n_classes):
    # Class weight calculation
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)
    n_samples = len(labels)

    weights = torch.FloatTensor([
        n_samples / (n_classes * class_counts.get(i, 1))
        for i in range(n_classes)
    ])

    print(f"\n Class distribution in training data: {dict(sorted(class_counts.items()))}")
    print(f"Class Weight: {weights.numpy()}")
    return weights

def create_weigthed_sampler(train_loader):
    """Sampler for Oversampling""" 
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)

    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

class AugmentedWrapper(Dataset):
    "Data Augmentation Wrapper"
    def __init__(self, dataset, augment_ratio=0.5, noise_std=0.15):
        self.dataset = dataset
        self.base_len = len(dataset)
        self.augment_len = int(self.base_len * augment_ratio)
        self.total_len = self.base_len + self.augment_len
        self.noise_std = noise_std

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.base_len:
            return self.dataset[idx]
        else:
            # Augmentation Sample
            base_idx = (idx - self.base_len) % self.base_len
            x, y = self.dataset[base_idx]
            noise = torch.randn_like(x) * self.noise_std
            return x + noise, y
        
# --------------------------------------------------------

# 8. Training and Evaluation

# --------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, device, epochs=30):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ------------------------------------------------------------

# 9. Main Execution

# -----------------------------------------------------------
def main():
    print("="*70)
    print("10th: Advanced Evaluation Metrics and Class Imbalance Handling")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ndevice: {device}")

    # Data Generation
    full_dataset = ImbalanceDataset(
        n_samples=2000, n_features=20, n_classes=4,
        imbalance_ratio=[0.5, 0.3, 0.15, 0.05]
    )

    # Distribution
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # ------------------------------------------------------------

    # Method 1: Weighted Loss

    # ---------------------------------------------------------------

    print("\n" + "="*70)
    print("Method 1 : Weight Loss")
    print("="*70)

    train_loader_normal = DataLoader(train_dataset, batch_size=64, shuffle=True)
    class_weights = get_class_weights(train_loader_normal, n_classes=4).to(device)

    model1 = MulitClassClassifier(20, 64, 4).to(device)
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

    print("\n[Start Training]")
    train_model(model1, train_loader_normal, criterion_weighted, optimizer1, device, 30)

    print("\n[Evaluation]")
    y_true, y_pred, y_proba = evaluate_model(model1, test_loader, device)

    print("\n1. Confustion Matrix")
    plot_confusion_matrix_with_analysis(y_true, y_pred, class_names)

    print("\n2. Detailed Metrics")
    calculate_detailed_metrics(y_true, y_pred, y_proba, class_names)

    print("\n3. ROC-AUC")
    plot_roc_curbes_multiclass(y_true, y_proba, class_names)

    # --------------------------------------------------------

    # Method 2: Oversampling

    # ------------------------------------------------------
    print("\n" + "="*70)
    print("Method 2: Oversampling")
    print("="*70)

    # Generate sampler after collecting labels from the regular loader
    temp_loader = DataLoader(train_dataset, batch_size=64, shuffle=False) #Temporary Dataloader 임시 데이터로더
    sampler = create_weigthed_sampler(temp_loader)
    train_loader_oversampled = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    model2 = MulitClassClassifier(20, 64, 4).to(device)
    criterion_normal = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

    print("\n[Start Training]")
    train_model(model2, train_loader_oversampled, criterion_normal, optimizer2, device, 30)

    print("\n[Evalution]")
    y_true2, y_pred2, y_proba2 = evaluate_model(model2, test_loader, device)

    print("\nConfusion Matrix")
    plot_confusion_matrix_with_analysis(y_true2, y_pred2, class_names)
    # Why? shuffle=False
    # That's because training objest is not shuffling order of data but collecting data obviously.
    # --------------------------------------------------------------
    
    # Method 3: Data Augmentation

    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("Method 3: Data Augmentation")
    print("="*70)

    augmented_dataset = AugmentedWrapper(train_dataset, augment_ratio=0.5, noise_std=0.15)
    train_loader_augmented = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

    model3 = MulitClassClassifier(20, 64, 4).to(device)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)

    print(f"\n Data size after augmentation: {len(augmented_dataset)} (raw dataset)")
    print("\n[Start Training]")
    train_model(model3, train_loader_augmented, criterion_normal, optimizer3, device, 30)

    print("\n[Evaluation]")
    y_true3, y_pred3, y_proba3 = evaluate_model(model3, test_loader, device)

    print("\nConfusion Matrix")
    plot_confusion_matrix_with_analysis(y_true3, y_pred3, class_names)

    # ----------------------------------------------------------------

    # Method 4: Temperature Scaling

    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print("4. Temperature Scaling")
    print("="*70)

    temp_scaler = TemperatureScaling().to(device)
    temperature = temp_scaler.calibrate(model1, test_loader, device)

    model1.eval()
    all_probs_before = []
    all_probs_after = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logtis = model1(inputs)

            probs_before = F.softmax(logtis, dim=1)
            probs_after = F.softmax(temp_scaler(logtis), dim=1)

            all_probs_before.extend(probs_before.cpu().numpy().tolist())
            all_probs_after.extend(probs_after.cpu().numpy().tolist())

    all_probs_before = np.array(all_probs_before)
    all_probs_after = np.array(all_probs_after)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    max_probs_before = all_probs_before.max(axis=1)
    max_probs_after = all_probs_after.max(axis=1)

    axes[0].hist(max_probs_before, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Maximum Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f"Before calibration (mean: {max_probs_before.mean():.3f}), fontsize=13")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(max_probs_after, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Maximum Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f"After calibration (mean: {max_probs_after.mean():.3f}, T={temperature:.3f}, fontsize=13)")
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------

    # Method5 : Method Comparison

    # --------------------------------------------------------------
    print("\n" + "="*70)
    print("5. Method Comparison")
    print("="*70)

    _, _, f1_1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_2, _ = precision_recall_fscore_support(y_true2, y_pred2, average='macro', zero_division=0)
    _, _, f1_3, _ = precision_recall_fscore_support(y_true3, y_pred3, average='macro', zero_division=0)

    methods = ["Weighted Loss", "Oversampling", "Augmentation"]
    scores = [f1_1,f1_2, f1_3]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, color=['skyblue', 'lightcoral', 'lightgreen'],
                  edgecolor='black', alpha=0.8)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_ylabel("Macro F1-Score", fontsiz=12)
    ax.set_title("Performance by Imbalance Handling Method", fontsize=14, pad=15)
    ax.set_ylim([0, max(scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nFinal Comparison")
    for method, score in zip(methods, scores):
        print(f"  {method:20s}: {score:.4f}")

    print("\n" + "="*70)
    print("Lecture Finished")
    print("="*70)

if __name__=="__main__":
    main()