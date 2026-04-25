import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             log_loss, brier_score_loss, precision_recall_fscore_support,
                             balanced_accuracy_score, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

def forward_selection(X, y, max_features=10):
    remaining_features = list(X.columns)
    selected_features = []
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    if len(X) > 20000:
        idx = np.random.choice(np.arange(len(X)), 20000, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
    else:
        X_sub = X
        y_sub = y

    for i in range(max_features):
        scores_with_candidates = []
        for candidate in remaining_features:
            features = selected_features + [candidate]
            X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_sub[features], y_sub, test_size=0.3, random_state=42)
            model.fit(X_train_fs, y_train_fs)
            scores_with_candidates.append((accuracy_score(y_test_fs, model.predict(X_test_fs)), candidate))
            
        scores_with_candidates.sort(reverse=True)
        best_candidate = scores_with_candidates[0][1]
        selected_features.append(best_candidate)
        remaining_features.remove(best_candidate)
        
    return selected_features

def draw_text_page(pdf, metrics_train, metrics_test, cls_report_train, cls_report_test, best_model_name):
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Title
    fig.text(0.5, 0.94, "PREVISAO DE COMISSOES %", fontsize=28, fontweight='bold', color='#2C3E50', ha='center')
    fig.text(0.5, 0.89, "Analise Completa - Classificadores Multi-Classe", fontsize=16, color='#7F8C8D', ha='center')
    
    # Model Badge
    fig.text(0.5, 0.82, f"MODELO VENCEDOR: {best_model_name.upper()}", fontsize=18, fontweight='bold', color='#E74C3C', ha='center',
            bbox=dict(facecolor='#FDEDEC', edgecolor='#E74C3C', boxstyle='round,pad=0.5'))
            
    # Draw KPI Matrix
    def draw_kpi(x, y, label, val_train, val_test, color, is_perc=True):
        fig.text(x, y, label, fontsize=12, color='#34495E', fontweight='bold', ha='center')
        fig.text(x-0.08, y-0.05, "TREINO", fontsize=9, color='#95A5A6', ha='center')
        val_tr_str = f"{val_train:.2%}" if is_perc else f"{val_train:.4f}"
        val_te_str = f"{val_test:.2%}" if is_perc else f"{val_test:.4f}"
        fig.text(x-0.08, y-0.1, val_tr_str, fontsize=16, color=color, fontweight='bold', ha='center')
        fig.text(x+0.08, y-0.05, "TESTE", fontsize=9, color='#95A5A6', ha='center')
        fig.text(x+0.08, y-0.1, val_te_str, fontsize=16, color=color, fontweight='bold', ha='center')

    draw_kpi(0.2, 0.65, "ACURACIA GLOBAL", metrics_train['acc'], metrics_test['acc'], '#2980B9')
    draw_kpi(0.5, 0.65, "ACURACIA BALANCEADA", metrics_train['bal_acc'], metrics_test['bal_acc'], '#16A085')
    draw_kpi(0.8, 0.65, "LOG-LOSS (CROSS-ENT)", metrics_train['logloss'], metrics_test['logloss'], '#C0392B', is_perc=False)
    draw_kpi(0.5, 0.45, "AVERAGE BRIER SCORE", metrics_train['brier'], metrics_test['brier'], '#8E44AD', is_perc=False)
    
    # Class Table Report manually
    fig.text(0.5, 0.3, "Detalhamento por Classe (TESTE):", fontsize=14, fontweight='bold', color='#34495E', ha='center')
    table_data = [["Classe", "Precision", "Recall", "F1-Score", "Suporte"]]
    for cls_name, vals in cls_report_test.items():
        if cls_name not in ['accuracy', 'macro avg', 'weighted avg']:
            table_data.append([cls_name, f"{vals['precision']:.3f}", f"{vals['recall']:.3f}", f"{vals['f1-score']:.3f}", str(vals['support'])])

    col_widths = [0.2, 0.15, 0.15, 0.15, 0.15]
    for i, row in enumerate(table_data):
        y_pos = 0.25 - (i * 0.04)
        for j, text in enumerate(row):
            weight = 'bold' if i == 0 else 'normal'
            x_pos = 0.15 + (j * 0.18)
            fig.text(x_pos, y_pos, text, fontsize=12, fontweight=weight, color='#2C3E50')

    pdf.savefig(fig)
    plt.close()

def plot_curves(y_test, y_score_test, le, pdf):
    classes = list(range(len(le.classes_)))
    y_test_bin = label_binarize(y_test, classes=classes)
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F1C40F']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#FFFFFF')
    
    # ROC Curve Multi-class OVR
    for i in range(len(classes)):
        class_name = le.inverse_transform([i])[0]
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_test[:, i])
        axes[0].plot(fpr, tpr, color=colors[i%len(colors)], lw=2, label=f'{class_name} (AUC: {auc(fpr, tpr):.2f})')
        
    axes[0].plot([0, 1], [0, 1], color='#BDC3C7', linestyle='--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_title('ROC Curve (One-vs-Rest) no TESTE', pad=15, fontweight='bold', color='#2C3E50')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle='-', alpha=0.3)
    
    # PR Curve Multi-class
    for i in range(len(classes)):
        class_name = le.inverse_transform([i])[0]
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score_test[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score_test[:, i])
        axes[1].plot(recall, precision, color=colors[i%len(colors)], lw=2, label=f'{class_name} (AP: {ap:.2f})')
        
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_title('Precision-Recall Curve (OVR) no TESTE', pad=15, fontweight='bold', color='#2C3E50')
    axes[1].set_xlabel('Recall (Sensibilidade)')
    axes[1].set_ylabel('Precision (Predicao Correta)')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle='-', alpha=0.3)
    
    pdf.savefig(fig)
    plt.close()

def plot_calibration_importance(best_m, X_train, y_test, y_score_test, le, pdf):
    classes = list(range(len(le.classes_)))
    y_test_bin = label_binarize(y_test, classes=classes)
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Calibration Curve (Reliability Diagram)
    for i in range(len(classes)):
        class_name = le.inverse_transform([i])[0]
        prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_score_test[:, i], n_bins=10)
        axes[0].plot(prob_pred, prob_true, marker='o', linewidth=1, color=colors[i%len(colors)], label=class_name)
        
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='#BDC3C7', label="Perfeitamente Calibrado")
    axes[0].set_ylabel("Frequencia Real")
    axes[0].set_xlabel("Probabilidade Prevista Média")
    axes[0].set_title('Reliability Diagram (Calibracao)', pad=15, fontweight='bold', color='#2C3E50')
    axes[0].legend()
    axes[0].grid(True, linestyle='-', alpha=0.3)
    
    # Feature Importance (Nativo - Random Forest via MDI)
    if hasattr(best_m, 'feature_importances_'):
        importances = best_m.feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10
        axes[1].barh(range(len(indices)), importances[indices], color='#8E44AD', align='center')
        axes[1].set_yticks(range(len(indices)))
        axes[1].set_yticklabels([X_train.columns[i] for i in indices])
        axes[1].set_xlabel('Score de Importância no Split (Gini)')
        axes[1].set_title('Top 10 Variaveis Importantes (Random Forest)', pad=15, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout(pad=3.0)
    pdf.savefig(fig)
    fig.savefig(IMAGES_DIR / "class_calib_imp.png", bbox_inches='tight')
    plt.close()

def compute_detailed_report(y_true, y_pred, y_prob, classes):
    from sklearn.metrics import classification_report
    rep_dict = classification_report(y_true, y_pred, output_dict=True, target_names=classes)
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    brier = np.mean([brier_score_loss(y_true_bin[:, i], y_prob[:, i]) for i in range(len(classes))])
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'logloss': log_loss(y_true, y_prob),
        'brier': brier
    }
    return metrics, rep_dict

def main():
    sns.set_theme(style="whitegrid", context="talk")
    print("Carregando dataset...")
    df = pd.read_parquet(DATA_DIR / 'BASE_processed.parquet')
    
    id_cols = ['apolice', 'apl_renovada1', 'item_renovado', 'nroapolicerefer', 'nroitemapolicerefer', 'cpf_cnpj']
    cols_to_exclude = set(id_cols + ['target_class', 'pct_comiss_depois'])
    feature_cols = [c for c in df.columns if c not in cols_to_exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]]
    
    X = df[feature_cols]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df['target_class']), name='target_class')
    
    best_features = forward_selection(X, y, max_features=15)
    X = X[best_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if len(X_train) > 40000:
        X_sub = X_train.sample(40000, random_state=42)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub, y_sub = X_train, y_train

    print("Treinando Modelo Random Forest...")
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                           {'n_estimators': [50], 'max_depth': [10]}, cv=3, n_jobs=-1)
    rf_grid.fit(X_sub, y_sub)
    best_m = RandomForestClassifier(**rf_grid.best_params_, random_state=42, class_weight='balanced', n_jobs=-1)
    best_m.fit(X_train, y_train)
    
    preds_tr = best_m.predict(X_train)
    probs_tr = best_m.predict_proba(X_train)
    preds_te = best_m.predict(X_test)
    probs_te = best_m.predict_proba(X_test)
    
    metrics_tr, rep_tr = compute_detailed_report(y_train, preds_tr, probs_tr, le.classes_)
    metrics_te, rep_te = compute_detailed_report(y_test, preds_te, probs_te, le.classes_)
    
    df['class_pred'] = le.inverse_transform(best_m.predict(X))
    df.to_parquet(DATA_DIR / 'BASE_with_class_preds.parquet', index=False)

    print("Gerando Apresentacao Analitica PDF (Classificacao)...")
    with PdfPages(REPORTS_DIR / 'Relatorio_Classificacao.pdf') as pdf:
        draw_text_page(pdf, metrics_tr, metrics_te, rep_tr, rep_te, "Random Forest")
        plot_curves(y_test, probs_te, le, pdf)
        plot_calibration_importance(best_m, X_train, y_test, probs_te, le, pdf)
        
        # Pagina 4: Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#FFFFFF')
        sns.heatmap(confusion_matrix(y_train, preds_tr), annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
        axes[0].set_title('Matriz de Confusao: TREINO')
        sns.heatmap(confusion_matrix(y_test, preds_te), annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
        axes[1].set_title('Matriz de Confusao: TESTE')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

if __name__ == "__main__":
    main()
