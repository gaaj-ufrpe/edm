import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

def iqr(x):
    return q3(x)-q1(x)

def upper_bound(x):
    return q3(x)+iqr(x)

def lower_bound(x):
    return q1(x)-iqr(x)

class Hyptest:
    
    def __init__(self, df, index_col, value_col):
        self.df = df
        self.index_col = index_col
        self.value_col = value_col

    def plot_probs(self):
        unique_index = self.df[self.index_col].unique()
        nrows=int(np.ceil(len(unique_index)/4))
        fig, ax = plt.subplots(nrows,4,figsize=(10,nrows*10))
        i = 1
        for idx in unique_index:
            irow=int(np.ceil(i/4))-1
            icol=((i-1)%4)
            axis = ax[irow,icol]
            stats.probplot(self.df[self.df[self.index_col] == idx][self.value_col], dist="norm", plot=axis)
            axis.set_title("Probability Plot - " +  idx)
            plt.margins(1.5)
            i=i+1
        plt.show()

    def normality_test_df(self):
        def test_by_index(index):
            values = self.df[self.df[self.index_col]==index][self.value_col]
            s1, p1 = stats.normaltest(values)
            s2, p2 = stats.shapiro(values)
            s3, p3 = stats.chisquare(values)
            return (p1,p1>.05,p2,p2>.05,p3,p3>.05)
        s_normality_df = pd.DataFrame(data=self.df[self.index_col].unique(), columns=[self.index_col])
        tmp = zip(*s_normality_df[self.index_col].apply(test_by_index))
        (s_normality_df['normaltest_pvalue'],s_normality_df['normaltest_normal'],
        s_normality_df['shapiro_pvalue'],s_normality_df['shapiro_normal'],
        s_normality_df['chisquare_pvalue'],s_normality_df['chisquare_normal']) = tmp
        return s_normality_df
    
    def normality_test(self):
        display(self.normality_test_df())
        self.plot_probs()

    def plot_distr(self):
        px.histogram(self.df,x=self.value_col,facet_col=self.index_col).show()
        px.violin(self.df,x=self.value_col,color=self.index_col).show()

    def remove_repeated(self,values):
        return list(dict.fromkeys(values))

    def make_commutative_pairs(self,values):
        xs = ys = values
        #combinaçao das listas exceto para os iguais o primeiro elemento da tupla sera semepre o menor
        #como a tupla é comutativa, deixa o primeiro menor, para remover depois com o dict.fromkeys
        pairs = [((x,y) if x<y else (y,x)) for x in xs for y in ys if x != y] 
        return self.remove_repeated(pairs)

    def test_samples_equality_df(self):
        def test_equality(series_a,series_b):
            values_a = self.df[self.df[self.index_col]==series_a][self.value_col]
            values_b = self.df[self.df[self.index_col]==series_b][self.value_col]
            s1, p1 = stats.ttest_ind(values_a,values_b)
            s2, p2 = stats.ks_2samp(values_a,values_b)
            s3, p3 = stats.mannwhitneyu(values_a,values_b)
            s4, p4 = stats.kruskal(values_a,values_b)
            return (p1,p1>.05,p2,p2>.05,p3,p3>.05,p4,p4>.05)

        pairs = self.make_commutative_pairs(self.df[self.index_col].unique())
        test_equality_df = pd.DataFrame(data=pairs,columns=['series_a','series_b'])
        tmp = test_equality_df.apply(lambda x: test_equality(x['series_a'],x['series_b']), axis=1)
        (test_equality_df['ttest_pvalue'],test_equality_df['ttest_equal'],
        test_equality_df['ks_pvalue'],test_equality_df['ks_equal'],
        test_equality_df['mannwhitneyu_pvalue'],test_equality_df['mannwhitneyu_equal'],
        test_equality_df['kruskal_pvalue'],test_equality_df['kruskal_equal']) = zip(*tmp)
        return test_equality_df
    
    def test_all(self):
        self.normality_test()
        self.plot_distr()
        display(self.test_samples_equality_df())
