#!/usr/bin/env python
# coding: utf-8

# <h5>Pacotes necessários

# In[3]:


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# <h5>Lendo o banco de dados

# In[4]:


dados = pd.read_csv('E:/Cursos/BootCamp Carrefour/Introdução a Ciência de Dados/train.csv')


# In[5]:


dados.head()


# <h5> Removendo variáveis que não são interessantes no nosso processo de modelagem

# In[6]:


dados = dados.drop(['Name','Ticket','Cabin', 'Embarked'], axis = 1)


# In[7]:


dados.head()


# <h5>Editando Chave e Variável de Resposta

# In[8]:


dados = dados.set_index(['PassengerId'])
dados = dados.rename(columns={'Survived':'target'},inplace = False )


# In[9]:


dados.head()


# <h5>Descritiva

# In[10]:


dados.describe()


# <h5>Descrição da variável qualitativa(sexo)

# In[11]:


dados.describe(include=['O'])


# <h5>Transformação dos dados

# Transformaremos as informações qualitativas em quantitativas
# 

# In[12]:


dados['Sex_F'] = np.where(dados['Sex'] == 'female', 1, 0)

dados['Pclass 1'] = np.where(dados['Pclass'] == 1, 1, 0)
dados['Pclass 2'] = np.where(dados['Pclass'] == 2, 1, 0)
dados['Pclass 3'] = np.where(dados['Pclass'] == 3, 1, 0)


# In[13]:


dados = dados.drop(['Pclass','Sex'], axis =1)


# In[14]:


dados.head()


# In[15]:


dados.isnull().sum() #Identificando valores não declarados


# In[16]:


dados.fillna(0, inplace = True) #Substituindo os valores não declarados por zero. Existem diversas abordagens para substitur, mas o zero foi escolhido


# In[17]:


dados.isnull().sum()


# <h5>Amostragem
#    

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(dados.drop(['target'], axis = 1),
                                                   dados['target'],
                                                    test_size = 0.3,
                                                    random_state = 1234)

[{'treino':x_train.shape},{'teste': x_test.shape}]
                                                    
                                                


# <h5>Modelo

# utilizaremos o modelo Random Forest

# In[26]:


rndforest = RandomForestClassifier(n_estimators = 1000,
                                  criterion = 'gini', #parâmetros do modelo. Olhar a documentação para outros
                                  max_depth = 5)

rndforest.fit(x_train, y_train)


# In[28]:


probabilidade = rndforest.predict_proba(dados.drop('target', axis = 1))[:,1]
classificacao = rndforest.predict(dados.drop('target', axis = 1))


# In[29]:


dados['probabilidade'] = probabilidade
dados['classificacao'] = classificacao


# In[30]:


dados

