import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("Iniciando a Análise do Check-in 01...")
print("-" * 50)

print("\nPasso 1: Carregando e limpando os dados...")

df = pd.read_csv('dados_governo.csv')

df['remuneracao'] = df['remuneraaao_contratual_r$'].str.replace(r'R\$', '', regex=True)
df['remuneracao'] = df['remuneracao'].str.replace(' ', '', regex=False)
df['remuneracao'] = df['remuneracao'].str.replace('.', '', regex=False)
df['remuneracao'] = df['remuneracao'].str.replace(',', '.', regex=False)

df['remuneracao'] = pd.to_numeric(df['remuneracao'], errors='coerce')

# Tratando valores nulos
df = df.dropna(subset=['remuneracao'])

# REGRA DO BOX-COX: Os dados precisam ser estritamente positivos (> 0)
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

print(f"Salários dentro de 1 Desvio Padrão: {porcentagem:.2f}% (Ideal próximo a 68%)")

print("\nPasso 3: Primeiros Testes de Hipóteses (KS e Shapiro)...")
print("-> REGRAS DA AULA PARA OS TESTES:")
print("   * Kolmogorov-Smirnov (KS): Recomendado para populações grandes (N > 30).")
print("   * Shapiro-Wilk (SW): Recomendado para amostras menores (4 a 2000 registros).")

resultado_ks = stats.kstest(df['remuneracao'], 'norm', args=(media, desvio_padrao))
p_value_ks = resultado_ks.pvalue

resultado_sw = stats.shapiro(df['remuneracao'])
p_value_sw = resultado_sw.pvalue

print(f"\nResultado do P-value Inicial (KS): {p_value_ks}")
print(f"Resultado do P-value Inicial (Shapiro): {p_value_sw}")

print("\nPasso 4: Transformação e Verificação Final...")

if p_value_ks > 0.05 or p_value_sw > 0.05:
    print("-> Status: A distribuição já É NORMAL! Nenhuma transformação é necessária.")
else:
    print("-> Status: A distribuição original NÃO É NORMAL (P-values menores que 0.05).")
    print("-> Ação: Aplicando a Transformação Inteligente (Box-Cox) vista na Aula 7...")
    
    # Aplicando o Box-Cox
    # Ele retorna a nova coluna normalizada e o Lambda ótimo que ele encontrou
    df['remuneracao_boxcox'], lambda_opt = stats.boxcox(df['remuneracao'])
    
    nova_media = df['remuneracao_boxcox'].mean()
    novo_desvio = df['remuneracao_boxcox'].std()
    
    novo_resultado_ks = stats.kstest(df['remuneracao_boxcox'], 'norm', args=(nova_media, novo_desvio))
    novo_p_value_ks = novo_resultado_ks.pvalue
    
    novo_resultado_sw = stats.shapiro(df['remuneracao_boxcox'])
    novo_p_value_sw = novo_resultado_sw.pvalue
    
    print(f"\n-> Lambda Ótimo calculado pelo Box-Cox: {lambda_opt:.4f}")
    print(f"-> NOVO P-value após Box-Cox (KS): {novo_p_value_ks}")
    print(f"-> NOVO P-value após Box-Cox (Shapiro): {novo_p_value_sw}")
    
    if novo_p_value_ks > 0.05 or novo_p_value_sw > 0.05:
        print("\n-> CONCLUSÃO FINAL: Sucesso! Após a transformação Box-Cox, os dados ficaram NORMAIS.")
    else:
        print("\n-> CONCLUSÃO FINAL: Os dados continuam não normais estatisticamente, mas a assimetria foi reduzida.")
        print("\n" + "="*50)
        print("POR QUE OS DADOS CONTINUAM NÃO NORMAIS? (Justificativa Técnica):")
        print("1. O rigor das amostras gigantes: temos mais de 17 mil linhas. O teste KS se torna extremamente")
        print("   rigoroso com amostras grandes, reprovando a normalidade por pequenas variações.")
        print("2. A regra do Shapiro: o teste de Shapiro-Wilk falha aqui porque nossa base ultrapassa muito")
        print("   o limite ideal ensinado na aula (que é de até 2000 registros).")
        print("3. A natureza do mundo real: a transformação Box-Cox (Lambda ótimo encontrado) reduz severamente")
        print("   a dispersão e a influência de salários outliers, porém, a estrutura de remuneração pública")
        print("   impede uma simetria matemática perfeita. Os dados estão prontos para uso em modelos.")
        print("="*50)

print("\nAnálise finalizada!")
print("-" * 50)

print("\nGerando gráficos... (Feche a janela do primeiro gráfico para ver o segundo)")

plt.figure(figsize=(10, 5))
plt.hist(df['remuneracao'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Média: R$ {media:.2f}')
plt.axvline(limite_baixo, color='green', linestyle='dashed', linewidth=2, label='-1 Desvio Padrão')
plt.axvline(limite_alto, color='green', linestyle='dashed', linewidth=2, label='+1 Desvio Padrão')
plt.title('Distribuição dos Salários e Desvio Padrão (Antes do Box-Cox)', fontsize=14)
plt.xlabel('Remuneração (R$)', fontsize=12)
plt.ylabel('Quantidade de Pessoas', fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
media_por_contrato = df.groupby('vanculo_empregatacio')['remuneracao'].mean().sort_values(ascending=False)
media_por_contrato.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Média Salarial por Tipo de Contrato', fontsize=14)
plt.xlabel('Tipo de Contrato', fontsize=12)
plt.ylabel('Média de Remuneração (R$)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
