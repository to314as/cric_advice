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
from sklearn.metrics import log_loss
import argparse
import warnings
import statsmodels as sm
from mord import LogisticAT
from mord import OrdinalRidge
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
            self.over_segments=np.array([0,6,11,14,17,20])
            self.wicket_segments=np.array([[0,0,0,0],[0,1,2,10],[0,2,4,10],[0,3,5,10],[0,4,6,10],[0,5,7,10],[0,5,7,10]])
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
        game_table = game_table[game_table.games_striker > 5]
        game_table = game_table[game_table.games_bowler > 3]
        game_table = game_table[game_table.extras <= 5]
        th = game_table[game_table.runs_off_bat == 3].iloc[0]
        fo = game_table[game_table.runs_off_bat == 4].iloc[0]
        fi = game_table[game_table.runs_off_bat == 5].iloc[0]
        si = game_table[game_table.runs_off_bat == 6].iloc[0]
        e5 = game_table[game_table.extras == 5].iloc[0]
        e4 = game_table[game_table.extras == 4].iloc[0]
        e3 = game_table[game_table.extras == 3].iloc[0]
        e2 = game_table[game_table.extras == 2].iloc[0]
        self.game_table=game_table
        self.add = pd.DataFrame()
        self.add = self.add.append([th, fo, fi, si, e2, e3, e4, e5])
        self.sizes=np.zeros((len(self.over_segments)-1,len(self.wicket_segments[0])-1))
        self.sizes2 = np.zeros((len(self.over_segments) - 1, len(self.wicket_segments[0]) - 1))
        self.os=[]
        self.ps=[]
        self.es=[]
        self.ess=[]
        self.os2=[]
        self.ps2=[]
        self.es2=[]
        self.ess2=[]


    def build_mo(self,y,c):
        #likelihood of a wicked being scored
        #m_o=LogisticRegression(random_state=0,penalty="l1",C=1,solver="saga",tol=0.0000001,max_iter=1000)
        m_o=LogisticRegressionCV()
        m_o=m_o.fit(c,y)
        print(m_o.score(c, y))
        return m_o

    def build_eo(self,y,c):
        #likelihood of a wicked being scored
        e_o=LogisticRegressionCV()
        e_o=e_o.fit(c,y)
        print(e_o.score(c, y))
        return e_o

    def build_eos(self,y,c):
        #likelihood of a wicked being scored
        e_os=LogisticRegressionCV(solver="lbfgs",multi_class="ovr",max_iter=1000)#LogisticAT(alpha=0)
        e_os=e_os.fit(c,y)
        print(e_os.score(c, y))
        return e_os

    def build_po(self,y,c):
        #likelihood of scoring r runs
        p_o=LogisticRegressionCV(solver="lbfgs",multi_class="ovr",max_iter=1000)#LogisticAT(alpha=0.5)
        p_o=p_o.fit(c,y)
        print(p_o.score(c, y))
        return p_o

    def train_inning(self,ini=1):
        if ini==1:
            var_e = ["balls_remain", "wickets_remain", "rr", "sr", "extras_striker", "extras_bowler"]
            var_w=["balls_remain", "wickets_remain", "rr", "sr", "e_r_career"]
            var_r=["balls_remain", "wickets_remain", "rr", "sr", "w_b"]
        else:
            var_e = ["balls_remain", "wickets_remain", "rrr", "sr", "extras_striker", "extras_bowler"]
            var_w = ["balls_remain", "wickets_remain", "rrr", "sr","e_r_career"]
            var_r = ["balls_remain", "wickets_remain", "rrr", "sr", "w_b"]
        df_segment = self.game_table[self.game_table.innings == ini]
        for i in range(1, len(self.over_segments)):
            df_segment_i = df_segment[self.over_segments[i - 1] <= df_segment.ball][df_segment.ball < self.over_segments[i]]
            for j in range(1, len(self.wicket_segments[i])):
                df_segment_ij = df_segment_i[self.wicket_segments[i][j - 1] <= self.w - df_segment_i.wickets_remain][
                    self.w - df_segment_i.wickets_remain < self.wicket_segments[i][j]]
                df_segment_ij = df_segment_ij.append(self.add)
                print(self.over_segments[i], self.wicket_segments[i][j], len(df_segment_ij))
                coefs = df_segment_ij[var_w]
                m_o = self.build_mo(df_segment_ij.wicket_scored, coefs)
                # print(m_o.coef_)
                coefs2 = df_segment_ij[var_r]
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                #coefs2 = poly.fit_transform(coefs2)
                p_o = self.build_po(df_segment_ij.runs_off_bat, coefs2)
                # print(p_o.coef_)
                coefs3 = df_segment_ij[var_e]
                e_o=self.build_eo(df_segment_ij.extras!=0, coefs3)
                coefs4 = df_segment_ij[df_segment_ij.extras!=0][var_e]
                e_os = self.build_eos(df_segment_ij[df_segment_ij.extras!=0].extras, coefs4)
                if ini==1:
                    self.os.append(m_o)
                    self.ps.append(p_o)
                    self.es.append(e_o)
                    self.ess.append(e_os)
                    self.sizes[i - 1, j - 1] = len(df_segment_ij)
                else:
                    self.os2.append(m_o)
                    self.ps2.append(p_o)
                    self.es2.append(e_o)
                    self.ess2.append(e_os)
                    self.sizes2[i - 1, j - 1] = len(df_segment_ij)
        self.adjust_coeff(ini)
        return

    def test_inning(self,ini):
        if ini==1:
            s=self.sizes
            os=self.os
            ps=self.ps
            es = self.es
            ess = self.ess
            var_e=["balls_remain", "wickets_remain", "rr", "sr","extras_striker","extras_bowler"]
            var_w=["balls_remain", "wickets_remain", "rr", "sr", "e_r_career"]
            var_r=["balls_remain", "wickets_remain", "rr", "sr", "w_b"]
        else:
            s=self.sizes2
            os=self.os2
            ps=self.ps2
            es = self.es2
            ess = self.ess2
            var_e = ["balls_remain", "wickets_remain", "rrr", "sr","extras_striker","extras_bowler"]
            var_w = ["balls_remain", "wickets_remain", "rrr", "sr", "e_r_career"]
            var_r = ["balls_remain", "wickets_remain", "rrr", "sr", "w_b"]
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
                coefs2 = df_segment_ij[var_r].values
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                #coefs2 = poly.fit_transform(coefs2)
                coefs3 = df_segment_ij[var_e].values
                coefs4 = df_segment_ij[df_segment_ij.extras!=0][var_e].values
                pred = ps[(i - 1) * 3 + (j - 1)].predict_proba(coefs2)
                pred_e_sole=es[(i - 1) * 3 + (j - 1)].predict_proba(coefs3)
                pred_e = ess[(i - 1) * 3 + (j - 1)].predict_proba(coefs4)
                base = [0 for i in range(len(np.unique(df_segment_ij.runs_off_bat)))]
                base[0] = 1
                base_e = [0 for i in range(len(np.unique(df_segment_ij.extras))-1)]
                base_e[0] = 1
                print(self.over_segments[i], self.wicket_segments[i][j],s[i - 1, j - 1])
                print("wicket prob:")
                print("Base:", log_loss(df_segment_ij.wicket_scored, [0 for i in range(len(df_segment_ij))]))
                print("Average:", log_loss(df_segment_ij.wicket_scored, [avg_probs_wicket for i in range(len(df_segment_ij))]))
                print("Model:",log_loss(df_segment_ij.wicket_scored, os[(i - 1) * 3 + (j - 1)].predict_proba(coefs)))

                print("extra probs")
                print("Base:", log_loss(df_segment_ij.extras!=0, [0 for i in range(len(df_segment_ij))]))
                print("Average:",log_loss(df_segment_ij.extras!=0, [avg_probs_extras_sole for i in range(len(df_segment_ij))]))
                print("Model:", log_loss(df_segment_ij.extras!=0, pred_e_sole))

                print("extra score")
                print("Base:", log_loss(df_segment_ij[df_segment_ij.extras!=0].extras, [base_e for i in range(len(df_segment_ij[df_segment_ij.extras!=0]))]))
                print("Average:",
                      log_loss(df_segment_ij[df_segment_ij.extras!=0].extras, [avg_probs_extras for i in range(len(df_segment_ij[df_segment_ij.extras!=0]))]))
                print("Model:", log_loss(df_segment_ij[df_segment_ij.extras!=0].extras, pred_e))

                print("runs prob:")
                print(ps[(i - 1) * 3 + (j - 1)].coef_)
                print((pred*np.array([[0,1,2,3,4,5,6] for i in range(len(pred))])).sum(axis=1))
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
            ess = self.ess
        else:
            s=self.sizes2
            os=self.os2
            ps=self.ps2
            es = self.es2
            ess = self.ess2
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
            os_sum=np.zeros(os[i*m+j].coef_.shape)
            ps_sum = np.zeros(ps[i*m+j].coef_.shape)
            es_sum = np.zeros(es[i*m+j].coef_.shape)
            ess_sum = np.zeros(ess[i*m+j].coef_.shape)
            k=[]
            for k in l[1][1]:
                os_sum+=os[(k[0]) * m + (k[1])].coef_
                ps_sum+=ps[(k[0]) * m + (k[1])].coef_
                es_sum+=es[(k[0]) * m + (k[1])].coef_
                ess_sum+=ess[(k[0]) * m + (k[1])].coef_
            if len(k) < 1:
                continue
            print(os_sum)
            print(ps_sum)
            os_sum /= n
            ps_sum /= n
            es_sum /= n
            ess_sum /= n
            self.os[i*m+j].coef_=self.os[i*m+j].coef_*w+(1-w)*os_sum
            self.ps[i*m+j].coef_ = self.ps[i*m+j].coef_ * w + (1 - w) * ps_sum
            self.es[i*m+j].coef_ = self.es[i*m+j].coef_ * w + (1 - w) * es_sum
            self.ess[i*m+j].coef_ = self.ess[i*m+j].coef_ * w + (1 - w) * ess_sum
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
        pickle.dump(self.ess, open(self.d + "ess.p", "wb"))
        pickle.dump(self.os2,open(self.d+"os2.p", "wb"))
        pickle.dump(self.ps2,open(self.d+"ps2.p", "wb"))
        pickle.dump(self.es2, open(self.d + "es2.p", "wb"))
        pickle.dump(self.ess2, open(self.d + "ess2.p", "wb"))
        pickle.dump(self.sizes, open(self.d + "sizes.p", "wb"))
        pickle.dump(self.sizes2, open(self.d + "sizes2.p", "wb"))

if __name__ == "__main__":
    model_trainer=ModelTrainer(d="./data/T20I/")
    model_trainer.train()