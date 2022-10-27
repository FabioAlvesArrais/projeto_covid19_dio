#!/usr/bin/env python
# coding: utf-8

# # Projeto covid19
# 
# ## Digital Innovation One
# 
# primeiro vamos importar alguma Bibliotecas

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


# Vamos importar os dados para o projeto
url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'


# In[3]:


df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
df


# In[4]:


# conferir tipos ou classes de cada coluna
df.dtypes


# nomes de colunas não devem ter letras maiusculas e nem caracteres especiais
# vamos implementar uma função para fazer a limpeza das colunas do dataframe

# In[5]:


import re

def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()


# In[6]:


corrige_colunas("ader/p ou") # teste


# In[7]:


# Vamos corrigir todas ascolunas do dataframe
df.columns = [corrige_colunas(col) for col in df.columns]


# In[8]:


df


# # Brasil
# ## trabalhando com Brasil, vamos selecionar apenas os dadosdo brasil para analisar

# In[9]:


df.countryregion.value_counts()


# In[10]:


df.countryregion.unique()


# In[11]:


df.loc[df.countryregion == 'Brazil']


# In[12]:


brasil = df.loc[
    (df.countryregion == 'Brazil') & (df.confirmed > 0)
]


# In[13]:


brasil


# # Casos confirmados

# In[14]:


# grafico da evolução dos casos confirmados
px.line(brasil, 'observationdate', 'confirmed', title= 'Casos confirmados no Brasil')


# In[15]:


# técnica de programação funciona
brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0])
))


# In[16]:


#visualizando
px.line(brasil, x='observationdate', y='novoscasos', title= 'Novos casos por dia')


# # Mortes

# In[17]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes',
              mode='lines+markers', line={'color':'red'})
)
#layout
fig.update_layout(title= 'Mortes por COVID-19 no Brasil')
fig.show()


# # taxa de crescimento
# taxa de crescimento = (presente/passado)**(1/n) - 1

# In[18]:


def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # se data Inicio for None define como a primeira data disponivel
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(Data_inicio)
        
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
        
    # agora nos vamos definir os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    # define o numero de pontos no tempo que vamos avaliar
    n = (data_fim - data_inicio).days
    
    #calcular a taxa
    taxa = (presente/passado)**(1/n) - 1
    
    return taxa*100
        


# In[19]:


# taxa de crescimento do COVID-19 no Brasil em todo o Periodo
taxa_crescimento(brasil, 'confirmed')


# In[20]:


# taxa de crescimento diario

def taxa_crescimento_diaria(data, variable, data_inicio=None):
    # se data Inicio for None define como a primeira data disponivel
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(Data_inicio)
        
    data_fim = data.observationdate.max()
            # define o numero de pontos no tempo que vamos avaliar
    n = (data_fim - data_inicio).days
    
    # Taxa calculada de um dia para outro
    taxas = list(map(
        lambda x:(data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1, n+1)
    ))
    return np.array(taxas) * 100


# In[21]:


tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')


# In[22]:


tx_dia


# In[23]:


primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:], 
        y=tx_dia, title= 'Taxa de crescimento de casos connfirmados no brasil'  )


# # Predições

# In[24]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# In[25]:


confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
confirmados


# In[26]:


res = seasonal_decompose(confirmados)


# In[27]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()


# # Arima

# In[32]:


get_ipython().system('pip install pmdarima')


# In[33]:


from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)


# In[34]:


fig = go.Figure(go.Scatter(
    x=confirmados.index, y= confirmados, name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'
))
fig.update_layout(title='Previsão de casos confirmados no brasil para os proximos 30 dias')


# # modelo de crescimento
# 
# vamos usar a biblioteca de fbprophet

# In[41]:


get_ipython().system('conda install -c conda-forge fbprophet -y')


# In[43]:


get_ipython().system('pip install pystan==2.19.1.1 prophet')


# In[44]:


import prophet


# In[45]:


get_ipython().system('conda install -c conda-forge fbprophet -y')


# In[46]:


from fbprophet import prophet


# In[ ]:




