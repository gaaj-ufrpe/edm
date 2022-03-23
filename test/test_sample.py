import sys
sys.path.append('./src')
import numpy as np
import pandas as pd
from odg import helper


def create_sample_df():
    np.random.seed = 12345
    n1a = pd.Series(np.random.normal(1, 1, 1000))
    n2a = pd.Series(np.random.normal(2, 1, 1000))
    e1a = pd.Series(np.random.exponential(1, 1000))
    e2a = pd.Series(np.random.exponential(2, 1000))
    np.random.seed = 23456
    n1b = pd.Series(np.random.normal(1, 1, 1000))
    n2b = pd.Series(np.random.normal(2, 1, 1000))
    e1b = pd.Series(np.random.exponential(1, 1000))
    e2b = pd.Series(np.random.exponential(2, 1000))

    s_df = pd.concat([n1a,n1b,n2a,n2b,e1a,e1b,e2a,e2b],keys=['n1a','n1b','n2a','n2b','e1a','e1b','e2a','e2b'])
    s_df = s_df.reset_index()
    s_df.columns = ['series', 'tmp', 'value']
    s_df.drop(columns=['tmp'],inplace=True)
    return s_df

s_df = create_sample_df()
htest = helper.Hyptest(s_df,'series','value')
htest.test_all()