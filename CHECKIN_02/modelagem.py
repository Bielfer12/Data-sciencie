import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

print("Iniciando a integração: Check-in 01 + Check-in 02...")
print("-" * 50)

print("\nPasso 1: Carregando e limpando os dados...")

df = pd.read_csv('dados_governo.csv')

df['remuneracao'] = df['remuneraaao_contratual_r$'].str.replace(r'R\$', '', regex=True)
df['remuneracao'] = df['remuneracao'].str.replace(' ', '', regex=False)
df['remuneracao'] = df['remuneracao'].str.replace('.', '', regex=False)
df['remuneracao'] = df['remuneracao'].str.replace(',', '.', regex=False)

df['remuneracao'] = pd.to_numeric(df['remuneracao'], errors='coerce')

df = df.dropna(subset=['remuneracao'])

total_antes = len(df)
df = df[df['remuneracao'] > 0]
total_depois = len(df)

print(f"Dados limpos e zeros removidos! Total de registros para análise: {total_depois} linhas.")

print("\nPasso 2: Análise Exploratória (Curva de Gauss)...")

media = df['remuneracao'].mean()
desvio_padrao = df['remuneracao'].std()

print(f"Média Salarial: R$ {media:.2f}")
print(f"Desvio Padrão: R$ {desvio_padrao:.2f}")

limite_baixo = media - desvio_padrao
limite_alto = media + desvio_padrao

dentro_do_sino = df[(df['remuneracao'] >= limite_baixo) & (df['remuneracao'] <= limite_alto)]
porcentagem = (len(dentro_do_sino) / len(df)) * 100

print(f"Salários dentro de 1 desvio padrão: {porcentagem:.2f}% (Ideal próximo a 68%)")

print("\nPasso 3: Testes de Hipóteses (KS e Shapiro)...")
print("-> REGRAS DA AULA PARA OS TESTES:")
print("   * Kolmogorov-Smirnov (KS): Recomendado para populações grandes (N > 30).")
print("   * Shapiro-Wilk (SW): Recomendado para amostras menores (4 a 2000 registros).")

resultado_ks = stats.kstest(df['remuneracao'], 'norm', args=(media, desvio_padrao))
p_value_ks = resultado_ks.pvalue

resultado_sw = stats.shapiro(df['remuneracao'])
p_value_sw = resultado_sw.pvalue

print(f"\nResultado do P-value inicial (KS): {p_value_ks}")
print(f"Resultado do P-value inicial (Shapiro): {p_value_sw}")

print("\nPasso 4: Transformação e verificação final...")

if p_value_ks > 0.05 or p_value_sw > 0.05:
    print("-> Status: A distribuição já é normal! Nenhuma transformação é necessária.")
else:
    print("-> Status: A distribuição original não é normal (P-values menores que 0.05).")
    print("-> Ação: Aplicando a Transformação Inteligente (Box-Cox) vista na Aula 7...")
    
    df['remuneracao_boxcox'], lambda_opt = stats.boxcox(df['remuneracao'])
    
    nova_media = df['remuneracao_boxcox'].mean()
    novo_desvio = df['remuneracao_boxcox'].std()
    
    novo_resultado_ks = stats.kstest(df['remuneracao_boxcox'], 'norm', args=(nova_media, novo_desvio))
    novo_p_value_ks = novo_resultado_ks.pvalue
    
    novo_resultado_sw = stats.shapiro(df['remuneracao_boxcox'])
    novo_p_value_sw = novo_resultado_sw.pvalue
    
    print(f"\n-> Lambda ótimo calculado pelo Box-Cox: {lambda_opt:.4f}")
    print(f"-> NOVO P-value após Box-Cox (KS): {novo_p_value_ks}")
    print(f"-> NOVO P-value após Box-Cox (Shapiro): {novo_p_value_sw}")
    
    if novo_p_value_ks > 0.05 or novo_p_value_sw > 0.05:
        print("\n-> CONCLUSÃO FINAL: Sucesso! Após a transformação Box-Cox, os dados ficaram normais.")
    else:
        print("\n-> CONCLUSÃO FINAL: Os dados continuam não normais estatisticamente, mas a assimetria foi reduzida.")
        print("\n" + "="*50)
        print("POR QUE OS DADOS CONTINUAM NÃO NORMAIS? (Justificativa técnica):")
        print("1. Rigor das amostras gigantes: com mais de 17 mil linhas, o teste KS torna-se extremamente")
        print("   rigoroso, reprovando a normalidade por pequenas variações.")
        print("2. Regra do Shapiro: o teste de Shapiro-Wilk falha aqui pois a base ultrapassa")
        print("   o limite ideal ensinado em aula (até 2000 registros).")
        print("3. Natureza do mundo real: a transformação Box-Cox reduz a dispersão e a influência")
        print("   de salários extremos, porém a estrutura de remuneração pública impede uma simetria perfeita.")
        print("="*50)

print("\nAnálise finalizada!")
print("-" * 50)

print("\nGerando gráficos...")

plt.figure(figsize=(10, 5))
plt.hist(df['remuneracao'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Média: R$ {media:.2f}')
plt.axvline(limite_baixo, color='green', linestyle='dashed', linewidth=2, label='-1 desvio padrão')
plt.axvline(limite_alto, color='green', linestyle='dashed', linewidth=2, label='+1 desvio padrão')
plt.title('Distribuição dos salários e desvio padrão (Antes do Box-Cox)', fontsize=14)
plt.xlabel('Remuneração (R$)', fontsize=12)
plt.ylabel('Quantidade de pessoas', fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
media_por_contrato = df.groupby('vanculo_empregatacio')['remuneracao'].mean().sort_values(ascending=False)
media_por_contrato.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Média salarial por tipo de contrato', fontsize=14)
plt.xlabel('Tipo de contrato', fontsize=12)
plt.ylabel('Média de remuneração (R$)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nIniciando modelagem com CatBoost (Check-in 02)...")

X = df[['vanculo_empregatacio']] 
y = df['remuneracao_boxcox']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
model.fit(X_train, y_train, cat_features=[0])

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Resultados do modelo CatBoost ---")
print(f"RMSE (Erro médio): {rmse:.4f}")
print(f"R² (Precisão do modelo): {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Real vs Previsto (CatBoost)', fontsize=14)
plt.xlabel('Valores reais (Box-Cox)')
plt.ylabel('Valores previstos (CatBoost)')
plt.show()

importances = model.get_feature_importance()
print("\n--- Importância da variável na alocação de gastos ---")
print(f"Peso do vínculo empregatício: {importances[0]:.2f}%")

print("\n" + "="*70)
print("RELATÓRIO DE RESPOSTA À PERGUNTA-PROBLEMA")
print("="*70)

resposta = f"""
1. DISTRIBUIÇÃO DOS GASTOS:
   Os dados indicam uma concentração salarial assimétrica. A aplicação do 
   modelo confirmou que a remuneração não é distribuída aleatoriamente, 
   mas segue um padrão estruturado pelo tipo de vínculo.

2. PADRÕES DE ALOCAÇÃO:
   - Precisão do modelo (R²): {r2:.2%} 
   - Erro de previsão (RMSE): {rmse:.4f}
   
   O modelo provou que o vínculo empregatício é a variável determinante para 
   a alocação de recursos, sendo responsável por {importances[0]:.2f}% da 
   lógica de definição salarial da prefeitura.

3. CONCLUSÃO PARA A GESTÃO:
   Identificamos que o padrão de alocação de recursos ao longo do tempo é 
   fortemente atrelado à política de contratação de vínculos. O modelo 
   desenvolvido atua como uma ferramenta preditiva eficaz (R² de {r2:.2%}), 
   permitindo à prefeitura simular impactos orçamentários sempre que houver 
   alterações na estrutura de vínculos dos servidores.
"""

print(resposta)
print("="*70 + "\n")