"""
Script auxiliar: exporta todas as imagens do Relatorio_Classificacao para a pasta reports/images/
Usa dados ja processados no data/BASE_processed.parquet
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             log_loss, brier_score_loss, balanced_accuracy_score,
                             precision_recall_curve, average_precision_score,
                             classification_report)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

print("Carregando dataset...")
df = pd.read_parquet(DATA_DIR / 'BASE_processed.parquet')

id_cols = ['apolice', 'apl_renovada1', 'item_renovado', 'nroapolicerefer', 'nroitemapolicerefer', 'cpf_cnpj']
cols_to_exclude = set(id_cols + ['target_class', 'pct_comiss_depois'])
feature_cols = [c for c in df.columns if c not in cols_to_exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]]

X = df[feature_cols]
le = LabelEncoder()
y = pd.Series(le.fit_transform(df['target_class']), name='target_class')

# Forward selection simplificado (top 15 features por importancia rapida)
def forward_selection(X, y, max_features=15):
    remaining_features = list(X.columns)
    selected_features = []
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    if len(X) > 20000:
        idx = np.random.choice(np.arange(len(X)), 20000, replace=False)
        X_sub = X.iloc[idx]; y_sub = y.iloc[idx]
    else:
        X_sub = X; y_sub = y
    for i in range(max_features):
        scores = []
        for candidate in remaining_features:
            feats = selected_features + [candidate]
            Xtr, Xte, ytr, yte = train_test_split(X_sub[feats], y_sub, test_size=0.3, random_state=42)
            model.fit(Xtr, ytr)
            scores.append((accuracy_score(yte, model.predict(Xte)), candidate))
        scores.sort(reverse=True)
        best = scores[0][1]
        selected_features.append(best)
        remaining_features.remove(best)
    return selected_features

print("Selecionando features...")
best_features = forward_selection(X, y, max_features=15)
X = X[best_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if len(X_train) > 40000:
    X_sub = X_train.sample(40000, random_state=42)
    y_sub = y_train.loc[X_sub.index]
else:
    X_sub, y_sub = X_train, y_train

print("Treinando Random Forest...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                       {'n_estimators': [50], 'max_depth': [10]}, cv=3, n_jobs=-1)
rf_grid.fit(X_sub, y_sub)
best_m = RandomForestClassifier(**rf_grid.best_params_, random_state=42, class_weight='balanced', n_jobs=-1)
best_m.fit(X_train, y_train)

preds_tr = best_m.predict(X_train)
probs_tr = best_m.predict_proba(X_train)
preds_te = best_m.predict(X_test)
probs_te = best_m.predict_proba(X_test)

classes = le.classes_
n_classes = len(classes)

# ───── Métricas ─────
def get_metrics(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    brier = np.mean([brier_score_loss(y_bin[:, i], y_prob[:, i]) for i in range(n_classes)])
    return {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'logloss': log_loss(y_true, y_prob),
        'brier': brier
    }

metrics_tr = get_metrics(y_train, preds_tr, probs_tr)
metrics_te = get_metrics(y_test, preds_te, probs_te)
rep_te = classification_report(y_test, preds_te, output_dict=True, target_names=classes)

# ───── IMAGEM 1: KPI Dashboard ─────
print("Gerando class_kpis.png...")
fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#F8F9FA')
fig.text(0.5, 0.95, "PREVISAO DE COMISSOES %", fontsize=28, fontweight='bold', color='#2C3E50', ha='center')
fig.text(0.5, 0.90, "Analise Completa - Classificadores Multi-Classe", fontsize=16, color='#7F8C8D', ha='center')
fig.text(0.5, 0.83, "MODELO VENCEDOR: RANDOM FOREST", fontsize=18, fontweight='bold', color='#E74C3C', ha='center',
         bbox=dict(facecolor='#FDEDEC', edgecolor='#E74C3C', boxstyle='round,pad=0.5'))

def draw_kpi(x, y, label, val_train, val_test, color, is_perc=True):
    fig.text(x, y, label, fontsize=12, color='#34495E', fontweight='bold', ha='center')
    fig.text(x-0.08, y-0.05, "TREINO", fontsize=9, color='#95A5A6', ha='center')
    v_tr = f"{val_train:.2%}" if is_perc else f"{val_train:.4f}"
    v_te = f"{val_test:.2%}" if is_perc else f"{val_test:.4f}"
    fig.text(x-0.08, y-0.10, v_tr, fontsize=18, color=color, fontweight='bold', ha='center')
    fig.text(x+0.08, y-0.05, "TESTE", fontsize=9, color='#95A5A6', ha='center')
    fig.text(x+0.08, y-0.10, v_te, fontsize=18, color=color, fontweight='bold', ha='center')

draw_kpi(0.20, 0.66, "ACURACIA GLOBAL",      metrics_tr['acc'],     metrics_te['acc'],     '#2980B9')
draw_kpi(0.50, 0.66, "ACURACIA BALANCEADA",  metrics_tr['bal_acc'], metrics_te['bal_acc'], '#16A085')
draw_kpi(0.80, 0.66, "LOG-LOSS (CROSS-ENT)", metrics_tr['logloss'], metrics_te['logloss'], '#C0392B', is_perc=False)
draw_kpi(0.50, 0.46, "AVERAGE BRIER SCORE",  metrics_tr['brier'],   metrics_te['brier'],   '#8E44AD', is_perc=False)

fig.text(0.5, 0.32, "Detalhamento por Classe (TESTE):", fontsize=13, fontweight='bold', color='#34495E', ha='center')
headers = ["Classe", "Precision", "Recall", "F1-Score", "Suporte"]
for j, h in enumerate(headers):
    fig.text(0.15 + j*0.18, 0.27, h, fontsize=11, fontweight='bold', color='#2C3E50')
for i, cls_name in enumerate([c for c in rep_te if c not in ['accuracy','macro avg','weighted avg']]):
    vals = rep_te[cls_name]
    row = [cls_name, f"{vals['precision']:.3f}", f"{vals['recall']:.3f}", f"{vals['f1-score']:.3f}", str(int(vals['support']))]
    for j, txt in enumerate(row):
        fig.text(0.15 + j*0.18, 0.22 - i*0.06, txt, fontsize=11, color='#2C3E50')

plt.tight_layout()
fig.savefig(IMAGES_DIR / "class_kpis.png", bbox_inches='tight', dpi=150)
plt.close()

# ───── IMAGEM 2: ROC + PR Curves ─────
print("Gerando class_roc_pr.png...")
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
colors = ['#3498DB', '#E74C3C', '#2ECC71']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#FFFFFF')

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_te[:, i])
    axes[0].plot(fpr, tpr, color=colors[i], lw=2, label=f'{classes[i]} (AUC: {auc(fpr, tpr):.2f})')
axes[0].plot([0, 1], [0, 1], color='#BDC3C7', linestyle='--', lw=2)
axes[0].set_title('ROC Curve (One-vs-Rest) no TESTE', pad=15, fontweight='bold', color='#2C3E50')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc="lower right"); axes[0].grid(True, linestyle='-', alpha=0.3)

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], probs_te[:, i])
    ap = average_precision_score(y_test_bin[:, i], probs_te[:, i])
    axes[1].plot(recall, precision, color=colors[i], lw=2, label=f'{classes[i]} (AP: {ap:.2f})')
axes[1].set_title('Precision-Recall Curve (OVR) no TESTE', pad=15, fontweight='bold', color='#2C3E50')
axes[1].set_xlabel('Recall (Sensibilidade)'); axes[1].set_ylabel('Precision (Predicao Correta)')
axes[1].legend(loc="lower left"); axes[1].grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
fig.savefig(IMAGES_DIR / "class_roc_pr.png", bbox_inches='tight', dpi=150)
plt.close()

# ───── IMAGEM 3: Calibration + Feature Importance (já existe, mas re-exporta para images/) ─────
print("Gerando class_calib_imp.png em images/...")
colors3 = ['#3498DB', '#E74C3C', '#2ECC71']
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#FFFFFF')

for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_test_bin[:, i], probs_te[:, i], n_bins=10)
    axes[0].plot(prob_pred, prob_true, marker='o', linewidth=1, color=colors3[i], label=classes[i])
axes[0].plot([0, 1], [0, 1], linestyle='--', color='#BDC3C7', label="Perfeitamente Calibrado")
axes[0].set_ylabel("Frequencia Real"); axes[0].set_xlabel("Probabilidade Prevista Média")
axes[0].set_title('Reliability Diagram (Calibracao)', pad=15, fontweight='bold', color='#2C3E50')
axes[0].legend(); axes[0].grid(True, linestyle='-', alpha=0.3)

importances = best_m.feature_importances_
indices = np.argsort(importances)[-10:]
axes[1].barh(range(len(indices)), importances[indices], color='#8E44AD', align='center')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([X_train.columns[i] for i in indices])
axes[1].set_xlabel('Score de Importância no Split (Gini)')
axes[1].set_title('Top 10 Variaveis Importantes (Random Forest)', pad=15, fontweight='bold', color='#2C3E50')

plt.tight_layout(pad=3.0)
fig.savefig(IMAGES_DIR / "class_calib_imp.png", bbox_inches='tight', dpi=150)
plt.close()

# ───── IMAGEM 4: Confusion Matrix ─────
print("Gerando class_confusion.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFFFFF')
sns.heatmap(confusion_matrix(y_train, preds_tr), annot=True, fmt='d', cmap='Blues',
            ax=axes[0], xticklabels=classes, yticklabels=classes, cbar=False)
axes[0].set_title('Matriz de Confusao: TREINO', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Previsto'); axes[0].set_ylabel('Real')
sns.heatmap(confusion_matrix(y_test, preds_te), annot=True, fmt='d', cmap='Oranges',
            ax=axes[1], xticklabels=classes, yticklabels=classes, cbar=False)
axes[1].set_title('Matriz de Confusao: TESTE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Previsto'); axes[1].set_ylabel('Real')
plt.tight_layout()
fig.savefig(IMAGES_DIR / "class_confusion.png", bbox_inches='tight', dpi=150)
plt.close()

print(f"✅ Todas as imagens exportadas com sucesso para {IMAGES_DIR}/")
print(f"  Métricas de Teste => Acurácia: {metrics_te['acc']:.2%} | Balanceada: {metrics_te['bal_acc']:.2%} | LogLoss: {metrics_te['logloss']:.4f} | Brier: {metrics_te['brier']:.4f}")
