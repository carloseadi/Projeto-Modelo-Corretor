import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

def draw_kpi_page(pdf, title, subtitle, metrics_train, metrics_test, model_name):
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#F4F6F6')

    fig.text(0.5, 0.94, title, fontsize=28, fontweight='bold', color='#212F3C', ha='center')
    fig.text(0.5, 0.89, subtitle, fontsize=16, color='#566573', ha='center')
    fig.text(0.5, 0.82, f"MODELO VENCEDOR: {model_name.upper()}", fontsize=18, fontweight='bold', color='#27AE60', ha='center',
            bbox=dict(facecolor='#E9F7EF', edgecolor='#27AE60', boxstyle='round,pad=0.5'))

    def draw_kpi(x, y, label, val_train, val_test, color):
        fig.text(x, y+0.02, label, fontsize=14, color='#2C3E50', fontweight='bold', ha='center')
        fig.text(x-0.08, y-0.03, "TREINO", fontsize=10, color='#95A5A6', ha='center')
        fig.text(x-0.08, y-0.08, f"{val_train:.3f}", fontsize=20, color=color, fontweight='bold', ha='center')
        fig.text(x+0.08, y-0.03, "TESTE", fontsize=10, color='#95A5A6', ha='center')
        fig.text(x+0.08, y-0.08, f"{val_test:.3f}", fontsize=20, color=color, fontweight='bold', ha='center')

    draw_kpi(0.25, 0.60, "R2 AJUSTADO", metrics_train['R2_adj'], metrics_test['R2_adj'], '#2980B9')
    draw_kpi(0.75, 0.60, "RMSE (Erro Quadratico)", metrics_train['RMSE'], metrics_test['RMSE'], '#C0392B')
    draw_kpi(0.25, 0.40, "MAE (Erro Absoluto)", metrics_train['MAE'], metrics_test['MAE'], '#F39C12')
    draw_kpi(0.75, 0.40, "MAPE (Erro Percentual %)", metrics_train['MAPE'], metrics_test['MAPE'], '#8E44AD')
    draw_kpi(0.50, 0.20, "RMSLE (Erro Logaritmo)", metrics_train['RMSLE'], metrics_test['RMSLE'], '#16A085')

    pdf.savefig(fig)
    fig.savefig(IMAGES_DIR / "reg_kpis.png", bbox_inches='tight')
    plt.close()

def compute_regression_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    r2_adj = 1 - (1-r2)*(n-1)/(n-n_features-1)
    
    # Avoid log of negative values 
    if np.any(y_true < 0) or np.any(y_pred < 0):
        rmsle = 0.0
    else:
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2_adj': r2_adj,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSLE': rmsle
    }

def main():
    sns.set_theme(style="whitegrid", context="talk")
    print("Carregando dataset de regressao...")
    df = pd.read_parquet(DATA_DIR / 'BASE_with_class_preds.parquet')
        
    id_cols = ['apolice', 'apl_renovada1', 'item_renovado', 'nroapolicerefer', 'nroitemapolicerefer', 'cpf_cnpj']
    cols_to_exclude = set(id_cols + ['target_class', 'pct_comiss_depois', 'class_pred'])
    feature_cols = [c for c in df.columns if c not in cols_to_exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]]
    
    X = df[feature_cols]
    y = df['pct_comiss_depois']
    
    # Adicionando uma segmentação por Premio original para plot depois
    X['Premio_Segment'] = pd.qcut(df['Premio_antes'], q=3, labels=['Baixo Ticket', 'Medio Ticket', 'Alto Ticket'])
    feature_cols_model = [c for c in X.columns if c != 'Premio_Segment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if len(X_train) > 40000:
        X_sub = X_train.sample(40000, random_state=42)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub, y_sub = X_train, y_train

    print("Treinando Ridge para Relatório...")
    ridge_grid = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0]}, cv=3, n_jobs=-1)
    ridge_grid.fit(X_sub[feature_cols_model], y_sub)
    best_m = Ridge(**ridge_grid.best_params_)
    best_m.fit(X_train[feature_cols_model], y_train)
    
    preds_train = best_m.predict(X_train[feature_cols_model]).clip(min=0)
    preds_test = best_m.predict(X_test[feature_cols_model]).clip(min=0)
    
    df['reg_pred'] = best_m.predict(df[feature_cols_model]).clip(min=0)
    df.to_parquet(DATA_DIR / 'BASE_final_preds.parquet', index=False)
    
    N_FEAT = len(feature_cols_model)
    metrics_tr = compute_regression_metrics(y_train, preds_train, N_FEAT)
    metrics_te = compute_regression_metrics(y_test, preds_test, N_FEAT)
    
    print("Gerando PDF Regressao Pitch Deck Profundo...")
    with PdfPages(REPORTS_DIR / 'Relatorio_Regressao.pdf') as pdf:
        # Pág 1: Resumo Parametrizado
        draw_kpi_page(pdf, "ESTIMATIVA DE COMISSOES %", "Resumo Executivo - Algoritmo de Regressao", metrics_tr, metrics_te, "Ridge Regression")
        
        # Pag 2: Hexbin ActualvsPredicted e Heatmap
        idx_te = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
        plot_df = pd.DataFrame({'Real': y_test.iloc[idx_te], 'Previsto': preds_test[idx_te]})
        g = sns.jointplot(data=plot_df, x="Real", y="Previsto", kind="hex", color="#8E44AD", marginal_kws=dict(bins=40, fill=True))
        lims = [max(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]), min(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
        g.ax_joint.plot(lims, lims, color='#E74C3C', linestyle='--', linewidth=2)
        g.fig.suptitle('Densidade: Previsto VS Real no Teste', y=1.05, fontsize=18, fontweight='bold')
        pdf.savefig(g.fig, bbox_inches='tight')
        g.fig.savefig(IMAGES_DIR / "reg_hexbin.png", bbox_inches='tight')
        plt.close(g.fig)
        
        # Pag 3: Analise de Residuos Fina (Q-Q e Scatter de Homocedasticidade)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#FFFFFF')
        res_test = y_test - preds_test
        
        # Q-Q Plot
        stats.probplot(res_test.iloc[idx_te], dist="norm", plot=axes[0])
        axes[0].set_title('Normalidade: Q-Q Plot dos Residuos')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Homoscedasticity Plot
        axes[1].scatter(preds_test[idx_te], res_test.iloc[idx_te], alpha=0.3, color='#F39C12')
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_title('Homocedasticidade: Residuos vs Valores Ajustados')
        axes[1].set_xlabel('Valor Previsto')
        axes[1].set_ylabel('Erro (Residuo)')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Pag 4: Erro por Segmento de Negocio
        fig, ax = plt.subplots(figsize=(10, 6))
        
        segment_df = pd.DataFrame({'Segmento': X_test['Premio_Segment'], 'ErroAbsoluto': np.abs(res_test)})
        mape_segment = segment_df.groupby('Segmento')['ErroAbsoluto'].mean().reset_index()
        
        sns.barplot(data=mape_segment, x='Segmento', y='ErroAbsoluto', palette='viridis', ax=ax)
        ax.set_title('MAE (Erro Absoluto) por Segmento de Volume de Premio (Teste)')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

if __name__ == "__main__":
    main()
