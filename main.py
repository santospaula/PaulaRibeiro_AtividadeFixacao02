# importando as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# fonte dataset
# https://www.kaggle.com/code/pedrodantas7/bodyfat-dataset

# carregando os dados do dataset
df =  pd.read_csv('heart_2020_cleaned.csv')

# explorando o cojunto de dados
df.shape
df.head()
df.tail()
df.info()

# são dezoito variáveis de base: HeartDisease (doença cardíaca), bmi (índice de massa corporal), 
# Smoking (fumante), AlcoholDriking (bebida alcoolica), stroke (AVC), PhysicalHealth (saúde física), 
# MentalHealth (saúde mental), DiffWalking (diferença caminhada), Sex (sexo), AgeCategory 
# (categoria idade), Race (corrida), Diabetic (diabetes), PhysicalActivity (atividade física), 
# GenHealth (saúde geral), SleepTime (hora de dormir), Asthma (asma), KidneyDisease (doença renal), 
# SkinCancer (câncer de pele)

# as variáveis são definidas da seguinte forma: 
# doença cardíaca: sim/não
# IMC: variável contínua
# tabaco: sim/não
# bebida alcoolica: sim/não
# saude fisica: variável contínua
# saude mental: variável contínua
# caminhada: sim/não
# sexo: feminino/masculino
# categoria idade: treze categorias (de 18 a 80 ou mais)
# raça: branca, preto, asiatico, índio americano/nativo do alasca, hispânica (latino-americanos que vivem nos EUA), outro
# diabetes: sim/não
# atividade fisica: sim/não
# saúde geral: muito bom, bom, excelente, justa, pobre
# horário de dormir: variável contínua
# asma: sim/não
# doença renal: sim/não
# câncer de pele: sim/não

# verifiquei que não tem Na, o que é um pouco estranho
# não ter valores ausentes, fui procurar por ele e 
# esse conjunto de dados já foi tratado previamente
print(df.isna().sum())

# renomear as colunas 
col_dic = {
    'HeartDisease':'DoencaCardiaca',
    'BMI':'IMC',
    'Smoking':'Tabaco',
    'AlcoholDrinking':'BebidaAlcoolica',
    'Stroke':'AVC',
    'PhysicalHealth':'SaudeFisica',
    'MentalHealth':'SaudeMental',
    'DiffWalking':'Caminhada',
    'Sex':'Sexo',
    'AgeCategory':'CategoriaIdade',
    'Race':'Raca',
    'Diabetic':'Diabetes',
    'PhysicalActivity':'AtividadeFisica',
    'GenHealth':'SaudeGeral',
    'SleepTime':'Sono',
    'Asthma':'Asma',
    'KidneyDisease':'DoencaRenal',
    'SkinCancer':'CancerPele'
}

df = df.rename(col_dic, axis=1)
df.head()

# Utilizar a função groupby para relacionar as principais causas da doença cardíaca
# As principais causas são: tabaco, obesidade (imc), álcool, diabetes, entre outros
# Primeiramente, uma nova coluna será criada, da seguinte forma:
# se faz uso de tabaco e/ou bebida alcoolica, receberá mau hábito
# se não faz uso de nenhum deles, receberá bom hábito
# Com isso, o objetivo é criar uma tabela de duas entradas, uma entrada
# será o hábitos e a outra doença cardiaca, para que possamos realizar
# um teste qui-quadrado e verificar se há associação entre as duas variáveis

def label_race(row):
  if row['Tabaco'] == 'Yes' and row['BebidaAlcoolica'] == 'Yes':
      return 'MauHabito'
  if row['Tabaco'] == 'Yes' and row['BebidaAlcoolica'] == 'No':
      return 'MauHabito'
  if row['Tabaco'] == 'No' and row['BebidaAlcoolica'] == 'Yes':
      return 'MauHabito'
  else:
    return 'BomHabito'


df.apply(label_race, axis=1)
df['Habitos'] = df.apply(label_race, axis=1)
df.head()
df.groupby(['DoencaCardiaca','Habitos']).size().unstack()

n11 = 168440
n12 = 11085
n21 = 123982
n22 = 16288

e11 = (23373*179525)/319795
e12 = (292422*140270)/319795
e21 = (23373*179525)/319795
e22 = (292422*140270)/319795

res11 = ((n11-e11)**2)/e11
res12 = ((n12-e12)**2)/e12
res21 = ((n21-e21)**2)/e21
res22 = ((n22-e22)**2)/e22

QuiQuadrado = res11 + res12 + res21 + res22
QuiQuadrado

# Conclusão: a 5% de significância, Xt (tabelado) = 3,841. 
# Como Xc (calculado) > Xt, rejeita-se Ho, logo podemos 
# concluir que as variáveis não são independentes.

# Diante disso, vamos filtrar em hábitos, se há predominância
# de mulheres ou homens que possuem bons/maus hábitos

df[(df['Sexo']=='Female') & (df['Habitos']=='MauHabito')].shape
propF = 67377/319795
print(f'A proporção de mulheres que têm maus hábitos, faz uso de tabaco e/ou de bebida alcoólica, é de {round(propF*100,2)}%.')

df[(df['Sexo']=='Male') & (df['Habitos']=='MauHabito')].shape
propM = 72893/319795
print(f'A proporção de homens que têm maus hábitos, faz uso de tabaco e/ou de bebida alcoólica, é de {round(propM*100,2)}%.')

# Uma outra variável que também é causa de doença cardíaca é a obesidade.
# Primeiramente, vamos categorizá-la, montar uma tabela de duas entradas
# e calcular a proporção de sobrepeso e obesidade em relação aos maus hábitos.

IMCnew = pd.cut(df['IMC'], bins=[0,18.5,24.9,29.9,100], labels=['BaixoPeso','PesoNormal','Sobrepeso','Obesidade'])
df['CategoricoIMC'] = IMCnew
df.head()
df.groupby(['Habitos','CategoricoIMC']).size().unstack()

print(f'A proporção de pessoas que estão em sobrepeso e obesidade, além de terem maus hábitos são {round(((51114+46869)/319795)*100,2)}%.') 

ax = df.groupby('Habitos')['IMC'].mean().plot(kind='bar')
ax.set_title('Média do índice de massa corporal para bom/mau hábitos', fontsize=12, pad=10)
ax.set_xlabel('Hábitos', fontsize=10, labelpad=10)
plt.xticks(rotation=0, horizontalalignment="center")
ax.set_ylabel('Média', fontsize=10, labelpad=10)

print(f'A média do índice de massa corporal daqueles que têm bom hábito é de 28.24 e dos que tem mau hábito é de 28.43.')

# Conclusão: As últimas análises foram feitos com objetivos explanatórios sobre o conjunto de dados. 
# Para verificar se há associação entre as variáveis é necessário realizar o teste estatístico apropriado.

# Por fim, exportaremos o conjunto de dados que teve acréscimo de duas colunas,
# sendo elas hábitos e categoria de IMC.

df.to_csv('DoencaCardiaca_Novo.csv')

