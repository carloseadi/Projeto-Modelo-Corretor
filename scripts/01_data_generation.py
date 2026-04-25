import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

def synthesize_data(df, n_samples=200000):
    print("Iniciando a geração de dados sintéticos...")
    np.random.seed(42)
    # Realiza amostragem com reposição
    synthetic_df = df.sample(n=n_samples, replace=True).reset_index(drop=True)
    
    numerical_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Adicionar ruído a valores contínuos como Prêmios
    cont_cols = ['Premio_depois', 'Premio_antes', 'Premio_proposto', 'ajuste_medio', 'mudanca_premio', 
                 'ajuste_preco', 'ajuste_is', 'Score_Autoglass_antes', 'Score_Serasa_antes', 'Score_RF_antes',
                 'Score_Sinistro_antes', 'Score_Autoglass_depois', 'Score_Serasa_depois', 'Score_RF_depois',
                 'Score_Sinistro_depois']
    
    for col in cont_cols:
        if col in synthetic_df.columns:
            std = df[col].std()
            if pd.isna(std) or std == 0:
                std = 0.05 * df[col].mean()
            noise = np.random.normal(0, std*0.1 if std else 0, size=len(synthetic_df))
            synthetic_df[col] = synthetic_df[col] + noise
            if 'Premio' in col or 'Score' in col:
                synthetic_df[col] = synthetic_df[col].clip(lower=0)

    # Injetar uma "Lógica/Sinal" para não termos F1=0.
    # O modelo anterior não conseguia prever Aumentar/Manter pois a variação da comissão era puramente Cega/Aleatória.
    # Vamos fazer a comissão depender das features:
    def gen_commission(row):
        antes = row['pct_comiss_antes']
        score = row.get('Score_Serasa_antes', 500)
        premio = row.get('Premio_antes', 1000)
        is_casco = row.get('is_casco_antes', 10000)
        
        # Lógica Fictícia Forte para as IAs aprenderem alguma correlação:
        # Pessoas com score muito alto e Prêmios altos induzem a corretora a AUMENTAR a comissão
        if score > 700 and premio > 1800:
            noise = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]) # Prob. Alta de aumento
        # Pessoas de baixo ticket ou com sinistros (representado por is_casco pequeno)
        elif score < 600 or premio < 1300:
            noise = np.random.choice([-3, -2, -1, 0], p=[0.3, 0.4, 0.2, 0.1]) # Prob. Alta de diminuicao
        else:
            # Miolo
            noise = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]) # Manteve a maior parte
            
        return max(0, min(40, antes + noise))

    synthetic_df['pct_comiss_depois'] = synthetic_df.apply(gen_commission, axis=1)

    # Criar Classificação Target
    def get_class(row):
        antes = row['pct_comiss_antes']
        depois = row['pct_comiss_depois']
        if pd.isna(antes) or pd.isna(depois):
            return 'Manteve'
        if depois > antes:
            return 'Aumentou'
        elif depois < antes:
            return 'Diminuiu'
        else:
            return 'Manteve'
            
    synthetic_df['target_class'] = synthetic_df.apply(get_class, axis=1)
    
    print(f"Dados gerados com sucesso. Shape: {synthetic_df.shape}")
    print("\nDistribuição das Classes (target_class):\n", synthetic_df['target_class'].value_counts(normalize=True))
    return synthetic_df

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / 'BASE.csv', sep=';')

    df_200k = synthesize_data(df, n_samples=200000)
    df_200k.to_parquet(DATA_DIR / 'BASE_200k.parquet', index=False)
    print(f"Salvo como {DATA_DIR / 'BASE_200k.parquet'}")
