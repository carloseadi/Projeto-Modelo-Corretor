import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

def ensembling_logic(row):
    obs_antes = row['pct_comiss_antes']
    previso = row['reg_pred']
    classe_predicao = row['class_pred']
    if pd.isna(obs_antes) or pd.isna(previso) or pd.isna(classe_predicao):
        return previso

    if classe_predicao == 'Manteve':
        return obs_antes
        
    reg_direction = 'Aumentou' if previso > obs_antes else ('Diminuiu' if previso < obs_antes else 'Manteve')
    
    if classe_predicao == 'Aumentou' and reg_direction == 'Diminuiu':
        return obs_antes - (previso - obs_antes)
        
    if classe_predicao == 'Diminuiu' and reg_direction == 'Aumentou':
        return obs_antes - (previso - obs_antes)
        
    return previso

def extract_ensemble_metrics(df):
    real = df['pct_comiss_depois']
    pred_reg = df['reg_pred']
    pred_ens = df['ensemble_pred']
    
    mae_reg = mean_absolute_error(real, pred_reg)
    mae_ens = mean_absolute_error(real, pred_ens)
    gain_mae = mae_reg - mae_ens
    
    df['reg_direction'] = np.where(pred_reg > df['pct_comiss_antes'], 'Aumentou', 
                                   np.where(pred_reg < df['pct_comiss_antes'], 'Diminuiu', 'Manteve'))
                                   
    discordancias = df[df['class_pred'] != df['reg_direction']]
    concordancias = df[df['class_pred'] == df['reg_direction']]
    taxa_concord = len(concordancias) / len(df)
    
    # Acerto na direção real
    df['real_direction'] = np.where(real > df['pct_comiss_antes'], 'Aumentou', 
                                   np.where(real < df['pct_comiss_antes'], 'Diminuiu', 'Manteve'))
                                   
    acc_dir_reg = np.mean(df['reg_direction'] == df['real_direction'])
    
    # Derivando a direção do ensemble:
    df['ens_direction'] = np.where(pred_ens > df['pct_comiss_antes'], 'Aumentou', 
                                   np.where(pred_ens < df['pct_comiss_antes'], 'Diminuiu', 'Manteve'))
    acc_dir_ens = np.mean(df['ens_direction'] == df['real_direction'])

    return {
        'taxa_concordancia': taxa_concord,
        'ganho_mae': gain_mae,
        'mae_reg': mae_reg,
        'mae_ens': mae_ens,
        'acc_dir_reg': acc_dir_reg,
        'acc_dir_ens': acc_dir_ens,
        'casos_discordantes': len(discordancias)
    }

def main():
    sns.set_theme(style="whitegrid", context="talk")
    print("Iniciando Consolidacao do Ensemble...")
    df = pd.read_parquet(DATA_DIR / 'BASE_final_preds.parquet')
    
    df['ensemble_pred'] = df.apply(ensembling_logic, axis=1).clip(lower=0)
    metrics = extract_ensemble_metrics(df)
    
    print("Gerando PDF Final Ensemble...")
    with PdfPages(REPORTS_DIR / 'Relatorio_Combinado_Ensemble.pdf') as pdf:
        # Pág 1: Métricas de Negócio - Comparação
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor('#F8F9FA')
        
        fig.text(0.5, 0.94, "AVALIACAO DE COMBINACAO (ENSEMBLE)", fontsize=26, fontweight='bold', color='#2C3E50', ha='center')
        fig.text(0.5, 0.89, "Resultados apos fusao das regras de Negocio sobre as NNs/Arvores", fontsize=15, color='#7F8C8D', ha='center')
        
        def draw_kpi(x, y, label, val_text, color):
            fig.text(x, y+0.05, label, fontsize=14, color='#34495E', fontweight='bold', ha='center')
            fig.text(x, y-0.05, val_text, fontsize=24, color=color, fontweight='bold', ha='center')
            
        draw_kpi(0.25, 0.70, "Taxa de Concordancia\n(Class x Reg)", f"{metrics['taxa_concordancia']:.2%}", '#2980B9')
        draw_kpi(0.75, 0.70, "Discordancias\n(Casos Corrigidos)", f"{metrics['casos_discordantes']}", '#E67E22')
        
        draw_kpi(0.25, 0.45, "Acuracia de Direcao\n(Apenas Regressao)", f"{metrics['acc_dir_reg']:.2%}", '#7F8C8D')
        draw_kpi(0.75, 0.45, "Acuracia de Direcao\n(Ensemble Final)", f"{metrics['acc_dir_ens']:.2%}", '#27AE60')
        
        draw_kpi(0.5, 0.2, "Ganho Liquido em MAE\n(Absoluto da Regressao menos Ensemble)", f"{metrics['ganho_mae']:+.4f} pp", 
                 '#27AE60' if metrics['ganho_mae'] > 0 else '#E74C3C')

        pdf.savefig(fig)
        plt.close()
        
        # Pag 2: Analise de Casos Submetidos à Discrepancia
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#FFFFFF')
        
        # Grafico 1: Barras de Comparação do Erro Total
        sns.barplot(x=['Regressao Pura', 'Modelo Ensemble'], y=[metrics['mae_reg'], metrics['mae_ens']], palette=['#E74C3C', '#27AE60'], ax=axes[0])
        axes[0].set_title('Erro Absoluto (MAE) Geral - Menor é Melhor', fontweight='bold')
        for i, val in enumerate([metrics['mae_reg'], metrics['mae_ens']]):
            axes[0].text(i, val/2, f"{val:.4f}", ha='center', color='white', fontweight='bold')
            
        # Grafico 2: Density Plot highlighting agreement vs disagreement
        idx_samp = np.random.choice(len(df), min(5000, len(df)), replace=False)
        sns.kdeplot(data=df.iloc[idx_samp], x='reg_pred', y='ensemble_pred', cmap="mako", fill=True, thresh=0.05, ax=axes[1])
        lims = [max(axes[1].get_xlim()[0], axes[1].get_ylim()[0]), min(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
        axes[1].plot(lims, lims, color='red', linestyle='--', label='Concordancia (Nao Corrigido)')
        axes[1].set_title('Superficie de Correcao: Regressao Original VS Ajuste via Regras', fontweight='bold')
        axes[1].set_xlabel('Previsao da Regressao Pura')
        axes[1].set_ylabel('Previsao do Ensemble (Corrigida)')
        axes[1].legend()

        plt.tight_layout()
        pdf.savefig(fig)
        fig.savefig(IMAGES_DIR / "ens_density.png", bbox_inches='tight')
        plt.close()

    df.to_parquet(DATA_DIR / 'BASE_OUTPUT_FINAL_PRONTA.parquet', index=False)
    print("Salvo com sucesso em 'Relatorio_Combinado_Ensemble.pdf'")
    
if __name__ == "__main__":
    main()
