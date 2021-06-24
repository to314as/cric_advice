from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as preprocessing
from sklearn.mixture import BayesianGaussianMixture
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import PolynomialFeatures
import pickle
from datetime import date,datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import statsmodels as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self,d="./data/allT20/",o=20,date=datetime(2015, 5, 7)):
        self.o=o
        self.w=10
        self.d=d
        self.metrics=['balls', 'games_striker', 'runs', 'dismissed', '4s', '6s', '50s', '100s', 's_r', 'Av', 'games_bowler', 'balls_bowl', 'concived_runs', 'wickets', 'noballs', 'wides', 'e_r','extras_bowler','extras_striker']
        if self.o==50: #odi matches
            self.over_segments=np.array([0,15,35,40,45,50])
            self.wicket_segments=np.array([[0,0,0,0],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]])
        else: #t20 matches
            self.over_segments=np.array([0,6,11,17,20])#14?
            self.wicket_segments=np.array([[0,0,0],[0,2,10],[0,4,10],[0,5,10],[0,7,10]])
        game_table = pd.read_csv(d + "stats/game_table.csv")
        game_table.start_date = pd.to_datetime(game_table.start_date, format='%Y-%m-%d')
        scaler = preprocessing.Normalizer()
        col=["balls_remain","rrr", "sr","extras_striker","extras_bowler","e_r_career","w_b","striker_performance"]
        game_table[col]=game_table[col].dropna()
        game_table[col] = scaler.fit_transform(game_table[col])
        #print(len(game_table))
        #game_table = game_table[game_table.start_date > date]
        game_table = game_table[game_table.rain==False]
        game_table = game_table[game_table.runs_off_bat != 7]
        game_table = game_table[game_table.games_striker > 3]
        game_table = game_table[game_table.games_bowler > 3]
        #game_table = game_table[game_table.batting_team.isin(["India","England","New Zealand","Australia","Pakistan","Netherlands"])]
        game_table = game_table[game_table.extras <= 5]
        th = game_table[game_table.runs_off_bat == 3].iloc[0]
        fo = game_table[game_table.runs_off_bat == 4].iloc[0]
        fi = game_table[game_table.runs_off_bat == 5].iloc[0]
        si = game_table[game_table.runs_off_bat == 6].iloc[0]
        wso=game_table[~game_table.wicket_scored].iloc[0]
        wso.wicket_scored=True
        eso = game_table[game_table.extras==0].iloc[0]
        eso.extras = 1
        eso2 = game_table[game_table.extras == 0].iloc[2]
        eso2.extras = 1
        self.game_table=game_table
        self.add = pd.DataFrame()
        self.add = self.add.append([th, fo, fi, si,wso,eso,eso2])
        self.sizes=np.zeros((len(self.over_segments)-1,len(self.wicket_segments[0])-1))
        self.sizes2 = np.zeros((len(self.over_segments) - 1, len(self.wicket_segments[0]) - 1))
        self.os=[]
        self.ps=[]
        self.es=[]
        self.os2=[]
        self.ps2=[]
        self.es2=[]
        self.var_e1 = ["balls_remain", "wickets_remain", "rr", "sr", "extras_striker", "extras_bowler"]
        self.var_w1 = ["balls_remain", "wickets_remain", "rr", "sr", "e_r_career"]
        self.var_r1 = ["balls_remain", "wickets_remain", "rr", "sr", "w_b","striker_performance"]
        self.var_e2 = ["balls_remain", "wickets_remain", "rrr", "sr", "extras_striker", "extras_bowler"]
        self.var_w2 = ["balls_remain", "wickets_remain", "rrr", "sr", "e_r_career"]
        self.var_r2 = ["balls_remain", "wickets_remain", "rrr", "sr", "w_b","striker_performance"]


    def build_mo(self,y,c):
        m_o = Logit(y, c)
        m_o = m_o.fit(method='lbfgs',maxiter=100,disp=False)
        return m_o

    def build_eo(self,y,c):
        e_o = Logit(y, c)
        e_o = e_o.fit(method='lbfgs',maxiter=100,disp=False)
        return e_o

    def build_po(self,y,c):
        # likelihood of scoring r runs
        p_o = OrderedModel(y, c)
        p_o = p_o.fit(method='lbfgs',maxiter=100,disp=False)
        # num_of_thresholds = 5
        # p_o.transform_threshold_params(p_o.params[-num_of_thresholds:])
        return p_o

    def train_inning(self,ini=1):
        if ini==1:
            var_e = self.var_e1
            var_w=self.var_w1
            var_r=self.var_r1
        else:
            var_e = self.var_e2
            var_w = self.var_w2
            var_r = self.var_r2
        df_segment = self.game_table[self.game_table.innings == ini]
        for i in range(1, len(self.over_segments)):
            df_segment_i = df_segment[self.over_segments[i - 1] <= df_segment.ball][df_segment.ball < self.over_segments[i]]
            for j in range(1, len(self.wicket_segments[i])):
                df_segment_ij = df_segment_i[self.wicket_segments[i][j - 1] <= self.w - df_segment_i.wickets_remain][
                    self.w - df_segment_i.wickets_remain < self.wicket_segments[i][j]]
                df_segment_ij = df_segment_ij.append(self.add)
                print(self.over_segments[i], self.wicket_segments[i][j], len(df_segment_ij))
                coefs = df_segment_ij[var_w]
                coefs = sm.tools.add_constant(coefs)
                m_o = self.build_mo(df_segment_ij.wicket_scored, coefs)
                # print(m_o.coef_)
                coefs2 = df_segment_ij[var_r]
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                #coefs2 = poly.fit_transform(coefs2)
                #coefs2 = sm.tools.add_constant(coefs2)
                p_o = self.build_po(df_segment_ij.runs_off_bat, coefs2)
                # print(p_o.coef_)
                coefs3 = df_segment_ij[var_e]
                coefs3 = sm.tools.add_constant(coefs3)
                e_o=self.build_eo(df_segment_ij.extras!=0, coefs3)
                coefs4 = df_segment_ij[df_segment_ij.extras!=0][var_e]
                if ini==1:
                    self.os.append(m_o)
                    self.ps.append(p_o)
                    self.es.append(e_o)
                    self.sizes[i - 1, j - 1] = len(df_segment_ij)
                else:
                    self.os2.append(m_o)
                    self.ps2.append(p_o)
                    self.es2.append(e_o)
                    self.sizes2[i - 1, j - 1] = len(df_segment_ij)
        self.adjust_coeff(ini)
        return

    def test_inning(self,ini):
        if ini==1:
            s=self.sizes
            os=self.os
            ps=self.ps
            es = self.es
            var_e = self.var_e1
            var_w = self.var_w1
            var_r = self.var_r1
        else:
            s=self.sizes2
            os=self.os2
            ps=self.ps2
            es = self.es2
            var_e = self.var_e2
            var_w = self.var_w2
            var_r = self.var_r2
        df_segment = self.game_table[self.game_table.innings == ini]
        for i in range(1, len(self.over_segments)):
            df_segment_i = df_segment[self.over_segments[i - 1] <= df_segment.ball][df_segment.ball < self.over_segments[i]]
            for j in range(1, len(self.wicket_segments[i])):
                df_segment_ij = df_segment_i[self.wicket_segments[i][j - 1] <= self.w - df_segment_i.wickets_remain][
                    self.w - df_segment_i.wickets_remain < self.wicket_segments[i][j]]
                df_segment_ij = df_segment_ij.append(self.add)
                avg_probs_wicket = df_segment_ij['wicket_scored'].value_counts(normalize=True).values
                avg_probs_runs = df_segment_ij['runs_off_bat'].value_counts(normalize=True).values
                avg_probs_extras = df_segment_ij['extras'].value_counts(normalize=True).values
                avg_probs_extras_sole = [avg_probs_extras[0],1-avg_probs_extras[0]]
                avg_probs_extras = df_segment_ij[df_segment_ij.extras != 0]['extras'].value_counts(normalize=True).values
                coefs = df_segment_ij[var_w].values
                coefs = sm.tools.add_constant(coefs)
                coefs2 = df_segment_ij[var_r].values
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                #coefs2 = poly.fit_transform(coefs2)
                #coefs2 = sm.tools.add_constant(coefs2)
                coefs3 = df_segment_ij[var_e].values
                coefs3 = sm.tools.add_constant(coefs3)
                coefs4 = df_segment_ij[df_segment_ij.extras!=0][var_e].values
                pred = ps[(i - 1) * 2 + (j - 1)].predict(coefs2)
                pred_e_sole=es[(i - 1) * 2 + (j - 1)].predict(coefs3)
                base = [0 for i in range(len(np.unique(df_segment_ij.runs_off_bat)))]
                base[0] = 1
                base_e = [0 for i in range(len(np.unique(df_segment_ij.extras))-1)]
                base_e[0] = 1
                print(self.over_segments[i], self.wicket_segments[i][j],s[i - 1, j - 1])
                print("wicket prob:")
                print(os[(i - 1) * 2 + (j - 1)].summary())
                print("Base:", log_loss(df_segment_ij.wicket_scored, [0 for i in range(len(df_segment_ij))]))
                print("Average:", log_loss(df_segment_ij.wicket_scored, [avg_probs_wicket for i in range(len(df_segment_ij))]))
                print("Model:",log_loss(df_segment_ij.wicket_scored, os[(i - 1) * 2 + (j - 1)].predict(coefs)))

                print("extra probs")
                print(es[(i - 1) * 2 + (j - 1)].summary())
                print("Base:", log_loss(df_segment_ij.extras!=0, [0 for i in range(len(df_segment_ij))]))
                print("Average:",log_loss(df_segment_ij.extras!=0, [avg_probs_extras_sole for i in range(len(df_segment_ij))]))
                print("Model:", log_loss(df_segment_ij.extras!=0, pred_e_sole))

                print("runs prob:")
                print(ps[(i - 1) * 2 + (j - 1)].summary())
                print("Base:", log_loss(df_segment_ij.runs_off_bat, [base for i in range(len(df_segment_ij))]))
                print("Average:", log_loss(df_segment_ij.runs_off_bat, [avg_probs_runs for i in range(len(df_segment_ij))]))
                print("Model:", log_loss(df_segment_ij.runs_off_bat, pred, labels=[0, 1, 2, 3, 4, 5, 6]))
                print()
                print("Base:", mean_squared_error(df_segment_ij.runs_off_bat, [(base*np.array([0,1,2,3,4,5,6])).sum() for i in range(len(df_segment_ij))])/len(pred))
                print("Average:",
                      mean_squared_error(df_segment_ij.runs_off_bat, [(avg_probs_runs*np.array([0,1,2,3,4,5,6])).sum() for i in range(len(df_segment_ij))])/len(pred))
                print("Model:", mean_squared_error(df_segment_ij.runs_off_bat, (pred*np.array([0,1,2,3,4,5,6])).sum(axis=1))/len(pred))



    def good_nb(self,s,i,j):
        g=0
        inds=[]
        if s[min(s.shape[0]-1,i+1),j]:
            g+=1
            inds.append((min(s.shape[0]-1,i+1),j))
        if s[max(0,i-1),j]:
            g+=1
            inds.append((max(0,i-1),j))
        if s[i,min(s.shape[1]-1,j+1)]:
            g+=1
            inds.append((i,min(s.shape[1]-1,j+1)))
        if s[i,max(0,j-1)]:
            g+=1
            inds.append((i,max(0,j-1)))
        return g,inds


    def adjust_coeff(self,ini,th=10000):
        if ini==1:
            s=self.sizes
            os=self.os
            ps=self.ps
            es = self.es
        else:
            s=self.sizes2
            os=self.os2
            ps=self.ps2
            es = self.es2
        low=s<th
        high=s>th
        lows={}
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if low[i,j]:
                    g,inds=self.good_nb(high,i,j)
                    lows[(i,j)]=[g,inds]
        lows=sorted(lows.items(), key=lambda x: x[1], reverse=True)
        print(lows)
        print(s)
        print(low)
        m=len(self.wicket_segments[0])-1
        for l in lows:
            #print(l)
            (i,j)=l[0]
            w=s[i,j]/th
            n=l[1][0]
            print(os[i*m+j].params.values)
            os_sum=np.zeros(os[i*m+j].params.values.shape)
            ps_sum = np.zeros(ps[i*m+j].params.values.shape)
            es_sum = np.zeros(es[i*m+j].params.values.shape)
            k=[]
            for k in l[1][1]:
                os_sum+=os[(k[0]) * m + (k[1])].params.values
                ps_sum+=ps[(k[0]) * m + (k[1])].params.values
                es_sum+=es[(k[0]) * m + (k[1])].params.values
            if len(k) < 1:
                continue
            print(os_sum)
            print(ps_sum)
            os_sum /= n
            ps_sum /= n
            es_sum /= n
            self.os[i*m+j].params= self.os[i*m+j].params*w+(1-w) * os_sum
            self.ps[i*m+j].params= self.ps[i*m+j].params * w + (1 - w) * ps_sum
            self.es[i*m+j].params= self.es[i*m+j].params * w + (1 - w) * es_sum
        return

    def train(self):
        self.train_inning(ini=1)
        self.test_inning(ini=1)
        print(self.sizes)
        self.train_inning(ini=2)
        self.test_inning(ini=2)
        print(self.sizes2)
        pickle.dump(self.os,open(self.d+"os.p", "wb"))
        pickle.dump(self.ps,open(self.d+"ps.p", "wb"))
        pickle.dump(self.es, open(self.d + "es.p", "wb"))
        pickle.dump(self.os2,open(self.d+"os2.p", "wb"))
        pickle.dump(self.ps2,open(self.d+"ps2.p", "wb"))
        pickle.dump(self.es2, open(self.d + "es2.p", "wb"))
        pickle.dump(self.sizes, open(self.d + "sizes.p", "wb"))
        pickle.dump(self.sizes2, open(self.d + "sizes2.p", "wb"))

if __name__ == "__main__":
    model_trainer=ModelTrainer(d="./data/T20I/")
    model_trainer.train()