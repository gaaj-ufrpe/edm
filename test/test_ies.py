import sys
sys.path.append('./src')
import numpy as np
import pandas as pd
from IPython.display import display
from odg import helper 

ies_df = pd.read_csv('./data/catadm-ies.csv',sep=':')
ies_df.columns = ['categoria','despesa_pessoal', 'receita', 'despesa_interna']
desp_df = ies_df[['categoria','despesa_pessoal']].copy(deep=True)
desp_df.replace(to_replace='Pública Federal', value='pub_fed', inplace=True)
desp_df.replace(to_replace='Privada sem fins lucrativos', value='priv_s_luc', inplace=True)
desp_df.replace(to_replace='Privada com fins lucrativos', value='priv_c_luc', inplace=True)
desp_df.replace(to_replace='Pública Estadual', value='pub_est', inplace=True)
desp_df.replace(to_replace='Pública Municipal', value='pub_mun', inplace=True)
desp_df.replace(to_replace='Especial', value='especial', inplace=True)
print('Dados:')
display(desp_df)

htest = helper.Hyptest(desp_df,'categoria','despesa_pessoal')
htest.test_all()

#Removendo os outliers
f = {'despesa_pessoal': ['median', 'std', helper.q1, helper.q3, helper.upper_bound, helper.lower_bound]}
agg_desp = desp_df.groupby(by=['categoria']).agg(f).reset_index()
agg_desp.columns = ['categoria','median', 'std', 'q1', 'q3', 'upper_bound', 'lower_bound']
print('Medidas por Categoria:')
display(agg_desp)

desp_df = desp_df.merge(agg_desp, on='categoria', how='left')
print('Dados Agregados com Medidas:')
display(desp_df)

desp_df = desp_df[(desp_df.despesa_pessoal>desp_df.lower_bound)&(desp_df.despesa_pessoal<desp_df.upper_bound)]
desp_df.rename(columns={'categoria':'categoria_s_outlier', 'despesa_pessoal':'despesa_pessoal_s_outlier'}, inplace=True)
display(desp_df)
htest = helper.Hyptest(desp_df,'categoria_s_outlier','despesa_pessoal_s_outlier')
htest.test_all()