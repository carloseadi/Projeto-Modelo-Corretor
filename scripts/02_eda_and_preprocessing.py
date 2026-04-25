import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import joblib

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

def main():
    print("Carregando BASE_200k.parquet...")
    df = pd.read_parquet(DATA_DIR / 'BASE_200k.parquet')

    print("\n--- PASSO 1: RELATÓRIO DE EDA ---")
    desc = df.describe(include='all').T
    desc.to_html(REPORTS_DIR / 'EDA_describe_report.html')
    print(f"Relatório de estatísticas descritivas salvo em {REPORTS_DIR / 'EDA_describe_report.html'}")

    print("\n--- PASSO 2: TRATAMENTO DE MISSINGS ---")
    # Identificar missing values
    missing_counts = df.isnull().sum()
    print("Valores ausentes por coluna:\n", missing_counts[missing_counts > 0])
    
    # Vamos separar as colunas por tipo
    id_cols = ['apolice', 'apl_renovada1', 'item_renovado', 'nroapolicerefer', 'nroitemapolicerefer', 'cpf_cnpj']
    target_class = ['target_class']
    target_reg = ['pct_comiss_depois']
    
    cols_to_exclude = set(id_cols + target_class + target_reg)
    features = [c for c in df.columns if c not in cols_to_exclude]
    
    num_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

    # Preenchimento de Missings
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    
    if len(num_features) > 0:
        df[num_features] = imputer_num.fit_transform(df[num_features])
    if len(cat_features) > 0:
        df[cat_features] = imputer_cat.fit_transform(df[cat_features])
        
    print("Missings tratados com sucesso.")

    print("\n--- PASSO 3: FEATURE ENGINEERING & PCA ---")
    # Engenharia de atributos
    if 'Premio_depois' in df.columns and 'Premio_antes' in df.columns:
        df['ratio_premio'] = np.where(df['Premio_antes'] == 0, 1, df['Premio_depois'] / df['Premio_antes'])
        num_features.append('ratio_premio')
    
    if 'Score_Serasa_depois' in df.columns and 'Score_Serasa_antes' in df.columns:
        df['diff_score_serasa'] = df['Score_Serasa_depois'] - df['Score_Serasa_antes']
        num_features.append('diff_score_serasa')

    # Scaling
    scaler = StandardScaler()
    df_num_scaled = scaler.fit_transform(df[num_features])
    
    # PCA para Redução / Criação de Componentes
    # Vamos criar 2 componentes principais para capturar a variação linear das features contínuas
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df_num_scaled)
    df['pca_comp1'] = pca_features[:, 0]
    df['pca_comp2'] = pca_features[:, 1]
    num_features.extend(['pca_comp1', 'pca_comp2'])
    print(f"PCA finalizado. Variância explicada pelas 2 comp: {sum(pca.explained_variance_ratio_):.2f}")

    print("\n--- PASSO 4: PREPARAÇÃO PARA MODELAGEM ---")
    # Encoding Variáveis Categóricas
    # Para performance e evitar maldição da dimensionalidade com muitos níveis (ex: cidades, corretores)
    # Vamos converter categorias para tipo 'category', que LightGBM e XGBoost (versões recentes) lidam bem nativamente
    # E vamos fazer LabelEncoding ou usar o tipo dummy apenas para o pipeline base (Logistic Regression).
    # Como não sabemos a cardinalidade, Target Encoding é melhor para cardinalidade alta, mas OHE servirá para modelos que precisam.
    # Em Pandas, get_dummies é rápido.
    df_prepared = pd.get_dummies(df, columns=cat_features, drop_first=True)
    
    # Salvar o pipeline / preprocessor (opcional mas boa prática, ignorado por simplicidade, 
    # dado que os modelos consumirão o df_prepared direto).
    
    print(f"Shape final do dataset preparado: {df_prepared.shape}")
    
    # Salvar base limpa pronta para a modelagem
    df_prepared.to_parquet(DATA_DIR / 'BASE_processed.parquet', index=False)
    print(f"Dataset salvo como {DATA_DIR / 'BASE_processed.parquet'}")

if __name__ == "__main__":
    main()
