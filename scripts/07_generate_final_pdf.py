"""
Gerador do RELATORIO_EXECUTIVO_FINAL.pdf
Usa reportlab para montar o PDF com texto formatado + imagens dos sub-relatórios
"""
import os
from PIL import Image as PILImage
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 HRFlowable, Table, TableStyle, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

# ─── Configurações de página ───
W, H = A4
MARGIN = 2.0 * cm
PAGE_WIDTH = W - 2 * MARGIN

doc = SimpleDocTemplate(
    str(REPORTS_DIR / "RELATORIO_EXECUTIVO_FINAL.pdf"),
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
    title="Relatorio Executivo Final - Previsao de Comissoes",
    author="Pipeline ML - Commission Predictor V2"
)

# ─── Estilos ───
styles = getSampleStyleSheet()

DARK_BLUE   = colors.HexColor('#1a3a6b')
MID_BLUE    = colors.HexColor('#2980B9')
RED         = colors.HexColor('#E74C3C')
GREEN       = colors.HexColor('#27AE60')
PURPLE      = colors.HexColor('#8E44AD')
ORANGE      = colors.HexColor('#F39C12')
TEAL        = colors.HexColor('#16A085')
LIGHT_BG    = colors.HexColor('#F0F4FF')
WHITE       = colors.white
GREY_TEXT   = colors.HexColor('#555555')
DARK_TEXT   = colors.HexColor('#2c2c2c')

def style(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

ST_TITLE = style('Title',
    fontSize=20, fontName='Helvetica-Bold',
    textColor=DARK_BLUE, leading=26, spaceAfter=4)

ST_SUBTITLE = style('Subtitle',
    fontSize=11, fontName='Helvetica',
    textColor=GREY_TEXT, leading=15, spaceAfter=12)

ST_H2 = style('H2',
    fontSize=14, fontName='Helvetica-Bold',
    textColor=DARK_BLUE, leading=20, spaceBefore=16, spaceAfter=4)

ST_H3 = style('H3',
    fontSize=11, fontName='Helvetica-Bold',
    textColor=MID_BLUE, leading=15, spaceBefore=10, spaceAfter=4)

ST_BODY = style('Body',
    fontSize=10, fontName='Helvetica',
    textColor=DARK_TEXT, leading=15, spaceAfter=8, alignment=TA_JUSTIFY)

ST_BULLET = style('Bullet',
    fontSize=10, fontName='Helvetica',
    textColor=DARK_TEXT, leading=14, spaceAfter=5,
    leftIndent=14, firstLineIndent=-10)

ST_CAPTION = style('Caption',
    fontSize=8.5, fontName='Helvetica-Oblique',
    textColor=GREY_TEXT, leading=12, spaceAfter=6, alignment=TA_CENTER)

ST_CENTER = style('Center',
    fontSize=10, fontName='Helvetica',
    textColor=DARK_TEXT, leading=14, alignment=TA_CENTER)

ST_WINNER = style('Winner',
    fontSize=13, fontName='Helvetica-Bold',
    textColor=RED, leading=18, alignment=TA_CENTER, spaceAfter=8)

ST_FINAL = style('Final',
    fontSize=10, fontName='Helvetica-Oblique',
    textColor=WHITE, leading=15, alignment=TA_CENTER)

def HR(color=DARK_BLUE, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=10, spaceBefore=10)

def img_block(path, caption, width_fraction=1.0, max_height_cm=None):
    """Retorna [Image, caption Paragraph] ou [] se arquivo não existir.
    Usa as dimensões reais da imagem para calcular o height correto."""
    if not os.path.exists(path):
        return []
    # Lê dimensões reais para preservar o aspect ratio
    pil_img = PILImage.open(path)
    px_w, px_h = pil_img.size
    ratio = px_h / px_w
    w = PAGE_WIDTH * width_fraction
    h = w * ratio
    # Limita altura máxima para não transbordar a página
    if max_height_cm is not None:
        h_max = max_height_cm * cm
        if h > h_max:
            h = h_max
            w = h / ratio
    img = Image(path, width=w, height=h)
    img.hAlign = 'CENTER'
    cap = Paragraph(caption, ST_CAPTION)
    return [img, cap, Spacer(1, 6)]

def section_header_table(num, title):
    """Cria um bloco de cabeçalho de seção com número em círculo + título."""
    data = [[Paragraph(f'<font color="white"><b>{num}</b></font>', ST_CENTER),
             Paragraph(f'<font color="#1a3a6b"><b>{title}</b></font>', ST_H2)]]
    t = Table(data, colWidths=[1.0*cm, PAGE_WIDTH - 1.0*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), DARK_BLUE),
        ('ALIGN',      (0,0), (0,0), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWPADDING', (0,0), (-1,-1), 6),
        ('BOX',        (0,0), (-1,-1), 0, colors.white),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    return t

def metrics_table_4(labels, tr_vals, te_vals, metric_colors):
    """Tabela 2×2 de KPIs com labels, Treino e Teste."""
    rows = []
    for i in range(0, len(labels), 2):
        pair = labels[i:i+2]
        row_labels = [Paragraph(f'<font color="{metric_colors[i]}"><b>{l}</b></font>', ST_CENTER) for l, _ in zip(pair, pair)]
        rows.append(row_labels + [''] * (2 - len(pair)))

        tr_row = [Paragraph(
            f'<font size="8" color="#aaaaaa">TREINO</font>  <font size="13" color="{metric_colors[i]}"><b>{tr_vals[i]}</b></font>'
            f'   <font size="8" color="#aaaaaa">TESTE</font>  <font size="13" color="{metric_colors[i]}"><b>{te_vals[i]}</b></font>',
            ST_CENTER)]
        if len(pair) == 2:
            tr_row.append(Paragraph(
                f'<font size="8" color="#aaaaaa">TREINO</font>  <font size="13" color="{metric_colors[i+1]}"><b>{tr_vals[i+1]}</b></font>'
                f'   <font size="8" color="#aaaaaa">TESTE</font>  <font size="13" color="{metric_colors[i+1]}"><b>{te_vals[i+1]}</b></font>',
                ST_CENTER))
        else:
            tr_row.append(Paragraph('', ST_CENTER))
        rows.append(tr_row)

    col_w = PAGE_WIDTH / 2
    t = Table(rows, colWidths=[col_w, col_w])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX',        (0,0), (-1,-1), 0.5, colors.HexColor('#dce4f5')),
        ('INNERGRID',  (0,0), (-1,-1), 0.3, colors.HexColor('#dce4f5')),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [LIGHT_BG, colors.HexColor('#e8eeff')]),
    ]))
    return t

def class_detail_table(rows_data):
    """Tabela de Precision/Recall/F1/Suporte por classe."""
    header = [Paragraph(f'<font color="white"><b>{h}</b></font>', ST_CENTER)
              for h in ['Classe', 'Precision', 'Recall', 'F1-Score', 'Suporte']]
    table_rows = [header]
    color_map = {'Aumentou': '#16A085', 'Diminuiu': '#16A085', 'Manteve': '#F39C12'}
    for cls, prec, rec, f1, sup in rows_data:
        c = color_map.get(cls, '#333')
        table_rows.append([
            Paragraph(f'<font color="#1a3a6b"><b>{cls}</b></font>', ST_CENTER),
            Paragraph(f'<font color="{c}"><b>{prec}</b></font>', ST_CENTER),
            Paragraph(f'<font color="{c}"><b>{rec}</b></font>', ST_CENTER),
            Paragraph(f'<font color="{c}"><b>{f1}</b></font>', ST_CENTER),
            Paragraph(sup, ST_CENTER),
        ])
    col_w = PAGE_WIDTH / 5
    t = Table(table_rows, colWidths=[col_w]*5)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8faff')]),
        ('BOX',        (0,0), (-1,-1), 0.5, colors.HexColor('#dce4f5')),
        ('INNERGRID',  (0,0), (-1,-1), 0.3, colors.HexColor('#dce4f5')),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    return t

def panel_header(text, bg_color=DARK_BLUE):
    data = [[Paragraph(f'<font color="white"><b>{text}</b></font>', ST_CENTER)]]
    t = Table(data, colWidths=[PAGE_WIDTH])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    return t

# ════════════════════════════════════════════
# CONSTRUÇÃO DO DOCUMENTO
# ════════════════════════════════════════════
story = []

# ── CAPA / CABEÇALHO ──
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Relatório Executivo Definitivo", ST_TITLE))
story.append(Paragraph("Modelo de Previsão de Comissões (End-to-End)", ST_TITLE))
story.append(Paragraph("Commission Predictor V2  •  200.000 perfis processados  •  Pipeline completo ML", ST_SUBTITLE))
story.append(HR(RED, 1.5))
story.append(Paragraph(
    "Este documento compila a interpretação corporativa e a defesa técnica do Pipeline de Aprendizado de Máquina "
    "construído para prever a elasticidade da margem de comissão de corretores de seguro em renovações. "
    "O arcabouço tecnológico processou <b>200.000 perfis sintético-baseados</b>, passando por motores de extração "
    "discriminatória, Random Forests, Penalizações por Ridge Regression e Heurísticas Especialistas de Ensemble.",
    ST_BODY))

story.append(HR())

# ══════════════════════════════════════════════
# SEÇÃO 1 — EDA & DATA PREP
# ══════════════════════════════════════════════
story.append(section_header_table("1", "Qualidade Preditiva e Saúde Variável (EDA & Data Prep)"))
story.append(Spacer(1, 6))

story.append(Paragraph("<b>Panorâmica Sanitária dos Dados:</b>", ST_H3))
story.append(Paragraph(
    "A base originária submetida apresentou imensa robustez, no entanto exibiu a carência natural e massiva de "
    "atributos terceiros (como falta de retornos nos <i>Scores do Serasa</i> e <i>Receita Federal</i>). "
    "Estes representaram ~115.000 missings estruturais da carteira.", ST_BODY))

story.append(Paragraph("<b>Como resolvemos e extraímos Qualidade:</b>", ST_H3))
story.append(Paragraph(
    "Através de medianas agrupadas e imputações com <b>Análise de Componentes Principais (PCA)</b> para features "
    "secundárias, blindamos a perda de sinal. Variáveis críticas sofreram <i>Feature Engineering</i>, originando "
    "matrizes de Diferença (Ex: <i>mudanca_premio</i>) e Scores Históricos Integrados — injetando Senso de Direção "
    "Preditiva na base.", ST_BODY))
story.append(Paragraph(
    "<b>Conclusão:</b> O dataset transmutou de uma matriz de 46 colunas ruidosas para um campo lapidado de "
    "<b>Top 15 Features super-preditivas</b>, eliminando o ruído matemático.", ST_BODY))

story.append(HR())

# ══════════════════════════════════════════════
# SEÇÃO 2 — CLASSIFICAÇÃO
# ══════════════════════════════════════════════
story.append(section_header_table("2", "Abordagem Passo 1: Classificação Direcional"))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>Decifrando se o Corretor \"Manteve\", \"Diminuiu\" ou \"Aumentou\"</i>", ST_CENTER))
story.append(Spacer(1, 6))

story.append(Paragraph("MODELO VENCEDOR: RANDOM FOREST", ST_WINNER))

story.append(Paragraph("<b>Dashboard de Métricas — KPIs da Classificação:</b>", ST_H3))
story.append(metrics_table_4(
    labels=["ACURÁCIA GLOBAL", "ACURÁCIA BALANCEADA", "LOG-LOSS (CROSS-ENT)", "AVERAGE BRIER SCORE"],
    tr_vals=["79,50%", "79,04%", "0.5399", "0.1034"],
    te_vals=["79,26%", "78,76%", "0.5562", "0.1066"],
    metric_colors=['#2980B9', '#16A085', '#C0392B', '#8E44AD']
))
story.append(Spacer(1, 8))

story.append(panel_header("📊  KPI Dashboard — Classificadores Multi-Classe (Random Forest)"))
story += img_block(str(IMAGES_DIR / "class_kpis.png"),
    "Painel executivo: Acurácia Global (~79%), Acurácia Balanceada, Log-Loss e Brier Score em Treino e Teste — com tabela de Precision / Recall / F1-Score por classe.",
    width_fraction=0.90, max_height_cm=12)

story.append(Paragraph("<b>Detalhamento por Classe no Conjunto de Teste:</b>", ST_H3))
story.append(class_detail_table([
    ("Aumentou", "0.896", "0.777", "0.833", "12.779"),
    ("Diminuiu", "0.900", "0.828", "0.862", "16.299"),
    ("Manteve",  "0.594", "0.758", "0.666", "10.922"),
]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "A classe <b>Manteve</b> apresenta F1-Score inferior (~0.67) — comportamento esperado em cenários de "
    "equilíbrio de preço. As classes <b>Aumentou</b> (F1: 0.83) e <b>Diminuiu</b> (F1: 0.86) demonstram "
    "alta assertividade, que é exatamente o valor de negócio essencial para o precificador.", ST_BODY))

story.append(Paragraph("<b>Curvas ROC (One-vs-Rest) e Precision-Recall:</b>", ST_H3))
story.append(Paragraph(
    "As curvas OVR comprovam o poder discriminatório do modelo em todas as três classes. AUC de "
    "<b>Aumentou (0.93)</b> e <b>Diminuiu (0.92)</b> estão próximos ao teto prático, enquanto "
    "<b>Manteve (0.78)</b> reflete a classe naturalmente mais difusa. "
    "A maior Average Precision na curva PR é <b>Diminuiu (AP: 0.86)</b>, seguida de <b>Aumentou (AP: 0.84)</b>.", ST_BODY))

story.append(panel_header("📈  ROC Curve (OVR) & Precision-Recall Curve — Teste"))
story += img_block(str(IMAGES_DIR / "class_roc_pr.png"),
    "Esquerda: ROC Curve OVR — AUC de 0.93 (Aumentou), 0.92 (Diminuiu) e 0.78 (Manteve). Todas superiores ao random-guess. "
    "Direita: Precision-Recall Curve — Average Precision de 0.84, 0.86 e 0.54 respectivamente.",
    width_fraction=1.0, max_height_cm=9)

story.append(Paragraph("<b>Calibração Probabilística e Importância de Variáveis:</b>", ST_H3))
story.append(Paragraph(
    "O <i>Reliability Diagram</i> demonstra que as probabilidades emitidas são honestas: quando o modelo afirma "
    "80% de chance de Aumentar, este evento se concretiza em ~80% dos casos reais. O gráfico de "
    "<i>Feature Importance</i> pelo Índice Gini revela que <b>Premio_antes</b> e <b>Score_Serasa_antes</b> "
    "são as variáveis mestras — evidenciando que o modelo reproduz o raciocínio do analista de negócios.", ST_BODY))

story.append(panel_header("🎯  Reliability Diagram (Calibração) & Top 10 Variáveis Importantes"))
story += img_block(str(IMAGES_DIR / "class_calib_imp.png"),
    "Esquerda: Curva de Calibração — alinhamento entre probabilidade prevista e frequência real por classe. "
    "Direita: Top 10 Variáveis por Score de Importância Gini — Premio_antes e Score_Serasa_antes lideram.",
    width_fraction=1.0, max_height_cm=9)

story.append(Paragraph("<b>Matrizes de Confusão — Treino e Teste:</b>", ST_H3))
story.append(Paragraph(
    "A diagonal dominante confirma baixíssima taxa de confusão entre classes opostas "
    "(<i>Aumentou ↔ Diminuiu</i>), que representaria o erro de maior custo de negócio. "
    "Os erros observados concentram-se na fronteira com <i>Manteve</i> — padrão esperado e aceitável.", ST_BODY))

story.append(panel_header("🔲  Matrizes de Confusão — Treino (Azul) & Teste (Laranja)", colors.HexColor('#7d3c00')))
story += img_block(str(IMAGES_DIR / "class_confusion.png"),
    "Diagonal dominante confirma alta concordância real-previsto. Erros mais frequentes ocorrem entre "
    "Manteve e as demais classes — padrão esperado pela natureza fuzzy da classe neutra.",
    width_fraction=1.0, max_height_cm=9)

story.append(HR())

# ══════════════════════════════════════════════
# SEÇÃO 3 — REGRESSÃO
# ══════════════════════════════════════════════
story.append(section_header_table("3", "Abordagem Passo 2: Regressão Percentual"))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>Computando o Valor Integral do Retorno Monetário / Margem %</i>", ST_CENTER))
story.append(Spacer(1, 6))

story.append(Paragraph(
    '<font color="#27AE60"><b>MODELO VENCEDOR: RIDGE REGRESSION</b></font>', ST_WINNER))

story.append(Paragraph("<b>Interpretação dos Indicadores:</b>", ST_H3))
bullets_reg = [
    "<b>R² Ajustado = 0.973:</b> A máquina justifica 97,3% das variações financeiras. Treino e Teste idênticos comprovam ausência de overfitting.",
    "<b>MAPE = 5,0% | MAE = 0.723:</b> Erros minimizados, entregando valores úteis ao Precificador sem sobressaltos extremos.",
    "<b>RMSE = 0.912 | RMSLE = 0.063:</b> Robustez confirmada para comissões de ticket alto, com erros logarítmicos sob controle.",
]
for b in bullets_reg:
    story.append(Paragraph(f"• {b}", ST_BULLET))
story.append(Spacer(1, 6))

story.append(panel_header("📈  KPIs do Modelo de Regressão — Ridge Regression (Modelo Vencedor)", GREEN))
story += img_block(str(IMAGES_DIR / "reg_kpis.png"),
    "Painel executivo com R² Ajustado (0.973), RMSE (0.912), MAE (0.723), MAPE (5,0%) e RMSLE (0.063) — Treino e Teste simétricos.",
    width_fraction=0.70, max_height_cm=12)

story.append(Paragraph("<b>Diagnóstico de Densidade — Previsto VS Real:</b>", ST_H3))
story.append(Paragraph(
    "O <i>Hexbin Scatter</i> de densidade aponta forte <b>Homocedasticidade</b>: a nuvem de pontos se "
    "mantém colada à diagonal de concordância perfeita em todo o range de comissões — de tickets baixos a "
    "altíssimos — sem alargamento de erros nas extremidades.", ST_BODY))

story.append(panel_header("🔬  Diagnóstico de Densidade — Previsto VS Real (Hexbin Scatter)", GREEN))
story += img_block(str(IMAGES_DIR / "reg_hexbin.png"),
    "Hexbin Scatter de densidade. Concentração central na diagonal confirma homocedasticidade e ausência de viés sistemático.",
    width_fraction=0.50, max_height_cm=12)

story.append(HR())

# ══════════════════════════════════════════════
# SEÇÃO 4 — ENSEMBLE
# ══════════════════════════════════════════════
story.append(section_header_table("4", "O Sistema Mestre Combinado: Lógica de Ensemble"))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<i>A Proteção Especialista Blindando as Máquinas</i>", ST_CENTER))
story.append(Spacer(1, 6))

story.append(Paragraph("<b>A Regra Interventiva:</b>", ST_H3))
story.append(Paragraph(
    "Se a IA Direcional classificou ser caso de 'Preço Mantido', e o Regressor calculasse mathematicamente "
    "+2.3% por ruído paramétrico — qual aceitar? <b>Cortamos o regressor e priorizamos a inteligência estrita!</b> "
    "Em caso de discordância grave (Classificador aumenta, Regressão cai), nossa equação penaliza a incerteza subtraindo a diferença.", ST_BODY))

story.append(Paragraph("<b>Resultado: MAE Ensemble vs Regressão Pura:</b>", ST_H3))
story.append(Paragraph(
    "O Ensemble corrigido reduziu o MAE de <b>0.7234</b> (Regressão Pura) para <b>0.6629</b> — uma "
    "<b>redução de ~8,4% no erro absoluto</b> pela aplicação das regras de negócio do precificador "
    "especialista sobre a saída bruta do modelo.", ST_BODY))

story.append(panel_header("⚖️  Distribuição dos Erros & Real vs Previsto — Ensemble vs Regressão Pura"))
story += img_block(str(IMAGES_DIR / "Final_Evaluation_Plot.png"),
    "Esquerda: Kernel Density dos Erros — Regressão Pura (vermelho) vs Ensemble corrigido (azul). "
    "Direita: Scatter Real vs Previsto — pontos do Ensemble agrupados ao redor da diagonal ideal.",
    width_fraction=1.0, max_height_cm=9)

story.append(panel_header("🎯  MAE Comparativo & Superfície de Correção do Ensemble"))
story += img_block(str(IMAGES_DIR / "ens_density.png"),
    "Esquerda: Comparativo MAE — Regressão Pura (0.7234) vs Ensemble (0.6629): redução de 8,4% no erro. "
    "Direita: Superfície de Correção mostrando o ajuste fino do Ensemble nos pontos de discordância.",
    width_fraction=1.0, max_height_cm=9)

story.append(HR())

# ══════════════════════════════════════════════
# SEÇÃO 5 — CONCLUSÕES
# ══════════════════════════════════════════════
story.append(section_header_table("5", "Defesa e Conclusões Preditivas"))
story.append(Spacer(1, 8))

conclusions = [
    ("<b>É Cientificamente Seguro:</b>",
     "A adoção maciça de Cross-Validation com Folds (CV=3) somada ao GridSearch Otimizado "
     "prova que os parâmetros foram alcançados matematicamente, não por testes empíricos engessados. A performance é duradoura."),
    ("<b>Alta Interpretabilidade:</b>",
     "Diferente de metodologias Black Box, os painéis executivos trazem Calibration Curves, "
     "Confusion Matrices, ROC Curves e Q-Q Plots. Sabemos precisamente por que uma apólice mudou de percentual — e com que confiança."),
    ("<b>Escalável Rapidamente:</b>",
     "Projetado em Parquet Files leves orientados a colunas. A pipeline empacota o resultado "
     "pronto para APIs, Streamlits e BI's — tornando-a em Produto finalizável em 1 semana ao invés de meses."),
]
for i, (label, text) in enumerate(conclusions, 1):
    story.append(Paragraph(f"{i}. {label} {text}", ST_BODY))

# Box final
final_bg_data = [[Paragraph(
    "O Sistema está inteiramente aprovado e preparado para implantação em Servidores Cloud.<br/>"
    "Esta arquitetura blindada promete estancar perdas de divergência e agregar estabilidade "
    "imediata sobre recotações massivas — escalando com precisão ao portfólio completo da companhia.",
    ST_FINAL)]]
final_t = Table(final_bg_data, colWidths=[PAGE_WIDTH])
final_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1a2540')),
    ('ROUNDEDCORNERS', [8]),
    ('TOPPADDING', (0,0), (-1,-1), 16),
    ('BOTTOMPADDING', (0,0), (-1,-1), 16),
    ('LEFTPADDING', (0,0), (-1,-1), 20),
    ('RIGHTPADDING', (0,0), (-1,-1), 20),
]))
story.append(Spacer(1, 8))
story.append(final_t)

# ── BUILD ──
doc.build(story)
print(f"PDF gerado: {REPORTS_DIR / 'RELATORIO_EXECUTIVO_FINAL.pdf'}")
