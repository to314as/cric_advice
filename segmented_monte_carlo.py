import pickle
import random
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import sklearn.linear_model
import sklearn.utils._weight_vector
warnings.filterwarnings("ignore")

# import os
# import time
# import pandas as pd
# import urllib.request
# import itertools
# from multiprocessing import Process, Pool
# from joblib import Parallel, delayed, parallel_backend
# import multiprocessing as mp
# from joblib import Parallel, delayed, parallel_backend
# from numpy import nan
# from collections import defaultdict
# import statsmodels as sm
# from statsmodels.miscmodels.ordinal_model import OrderedModel
# from statsmodels.discrete.discrete_model import Probit
# import matplotlib.pyplot as plt
# from player_stats import PlayerStats

class SegmentedMonteCarlo:
    def __init__(self,d="./data/allT20/",num_iter=1000):
        self.d=d
        #self.over_segments_odi=[0,15,35,40,45,50]
        #self.wicket_segments_odi=[[0,0,0,0],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]]
        self.over_segments=[0,6,11,15,18,20]
        self.wicket_segments=[[0,0,0,0],[0,2,4,10],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]]
        self.ov=20
        self.sizes = pickle.load(open(self.d + "sizes.p", "rb"))
        self.sizes2 = pickle.load(open(self.d + "sizes2.p", "rb"))
        self.os = pickle.load(open(self.d + "os.p", "rb"))
        self.ps = pickle.load(open(self.d + "ps.p", "rb"))
        self.es = pickle.load(open(self.d + "es.p", "rb"))
        self.ess = pickle.load(open(self.d + "ess.p", "rb"))
        self.os2 = pickle.load(open(self.d + "os2.p", "rb"))
        self.ps2 = pickle.load(open(self.d + "ps2.p", "rb"))
        self.es2 = pickle.load(open(self.d + "es2.p", "rb"))
        self.ess2 = pickle.load(open(self.d + "ess2.p", "rb"))
        self.num_iter=num_iter

    def change_bowler(self,b,i):
        if (self.ov*6-b)%18==0 and b!=self.ov*6:
            return i+1
        return i

    def O(self,b,w,rr,s,w_b=20,e_r=7,seg=0,ini=1):
        if ini==1:
            m_o=self.os[seg]
        else:
            m_o=self.os2[seg]
        #return m_o.predict([b,w,rr,s,w_b,e_r])
        return m_o.predict([b, w, rr, s, e_r])

    def P(self,b,w,rr,s,w_b=20,avg=15,s_r=95,seg=0,ini=1):
        if ini==1:
            p_o=self.ps[seg]
        else:
            p_o=self.ps2[seg]
        #return p_o.predict([b,w,rr,s,w_b,avg,s_r])
        return p_o.predict([b, w, rr, s, w_b])

    def E(self,b,w,rr,s,extras_striker=0,extras_bowler=0,seg=0,ini=1):
        if ini==1:
            e_o=self.es[seg]
        else:
            e_o=self.es2[seg]
        return e_o.predict([b,w,rr,s,extras_striker,extras_bowler])

    def ES(self,b,w,rr,s,extras_striker=0,extras_bowler=0,seg=0,ini=1):
        if ini==1:
            e_os=self.ess[seg]
        else:
            e_os=self.ess2[seg]
        return e_os.predict([b,w,rr,s,extras_striker,extras_bowler])

    def monte_carlo(self,r,b,w,rr,so,st,players_wb,players_er,players_sr,players_av,extras_strikers,extras_bowlers,ini=1,seg=5):
        if ini==1:
            s_model=self.sizes
            os_model=self.os
            ps_model=self.ps
            es_model = self.es
            ess_model = self.ess
        else:
            s_model=self.sizes2
            os_model=self.os2
            ps_model=self.ps2
            es_model = self.es2
            ess_model = self.ess2
        f=1
        p_striker=int(10-w)
        p_partner=p_striker+1
        next_striker=p_partner
        bowler=int((self.ov*6-b)//18)
        extras=0
        k=self.over_segments[seg]
        #print(players_wb)
        #print(players_er)
        con=True
        while con:
            #print(b)
            rand=np.random.rand()
            seg1=0
            seg2=0
            er=players_er[bowler]
            wb=players_wb[p_striker*f+p_partner*(1-f)]
            av=players_av[p_striker*f+p_partner*(1-f)]
            sr=players_sr[p_striker*f+p_partner*(1-f)]
            e_s=extras_strikers[p_striker*f+p_partner*(1-f)]
            eb=extras_bowlers[bowler]
            rrr = r / b
            for i in range(1,len(self.over_segments)):
                if (self.ov*6-b)//6>=self.over_segments[i]:
                    seg1+=1
                else:
                    break
            for i in range(1,len(self.wicket_segments[seg1])):
                if (10-w)>=self.wicket_segments[seg1][i]:
                    seg2+=1
                else:
                    break
            seg=(seg1-1)*3+seg2
            #print(b, w, rr, f * so + (1 - f) * st, e_s, eb)
            if ini==1:
                e_o=es_model[seg]
                e = e_o.predict_proba([[b, w, rr, f * so + (1 - f) * st, e_s, eb]])[0]
                #print(e)
                e_os=ess_model[seg]
                es=e_os.predict_proba([[b, w, rr, f * so + (1 - f) * st, e_s, eb]])[0]
                m_o=os_model[seg]
                #o=m_o.predict_proba([[b,w,rr,f*so+(1-f)*st,wb,er]])[0]
                o = m_o.predict_proba([[b, w, rr, f * so + (1 - f) * st, er]])[0]
                #print(o)
                p_o=ps_model[seg]
                #p=p_o.predict_proba([[b,w,rr,f*so+(1-f)*st,wb,av,sr]])[0]
                p = p_o.predict_proba([[b, w, rr, f * so + (1 - f) * st, wb]])[0]
                #print(p)
            else:
                e_o = es_model[seg]
                e = e_o.predict_proba([[b, w, rrr, f * so + (1 - f) * st, e_s, eb]])[0]
                # print(e)
                e_os = ess_model[seg]
                es = e_os.predict_proba([[b, w, rrr, f * so + (1 - f) * st, e_s, eb]])[0]
                m_o = os_model[seg]
                #o = m_o.predict_proba([[b, w, rrr, f * so + (1 - f) * st, wb, er]])[0]
                o = m_o.predict_proba([[b, w, rrr, f * so + (1 - f) * st, er]])[0]
                # print(o)
                p_o = ps_model[seg]
                #p = p_o.predict_proba([[b, w, rrr, f * so + (1 - f) * st, wb, av, sr]])[0]
                p = p_o.predict_proba([[b, w, rrr, f * so + (1 - f) * st, wb]])[0]
                # print(p)
            if rand<e[1]:
                #print(e)
                ex = 1#random.choices([i for i in range(1,len(es)+1)], weights=es)[0]
                extras+=ex
                #print("extras",ex)
                if ini==1:
                    r += ex
                else:
                    r -= ex
                rr=(rr*(self.ov*6-b)/6+ex)*(6/(self.ov*6-b+1))
            elif rand<e[1]+e[0]*o[1]:
                b-=1
                w-=1
                so-=f*so
                st-=(1-f)*st
                rr=(rr*(self.ov*6-b)/6+0)*(6/(self.ov*6-b+1))
                next_striker+=1
                if f==1:
                    p_striker=next_striker
                else:
                    p_partner=next_striker
                f=(1-f)
            else:
                p=random.choices([0,1,2,3,4,5,6], weights=p)[0]
                if ini == 1:
                    r += p
                else:
                    r -= p
                rr=(rr*(self.ov*6-b)/6+p)*(6/(self.ov*6-b+1))
                b-=1
                so+=f*p
                st+=(1-f)*p
                if p%2==1 and b%6!=0:
                    f=(1-f)
                if p%2==0 and b%6==0:
                    f=(1-f)
                bowler=self.change_bowler(b,bowler)
            if ini==1:
                con=(b-((self.ov-k)*6)>0 and w>0)
            else:
                con=(b-((self.ov-k)*6)>0 and w>0 and r>=0)
        return r

    def simulate_segment(self,stats,probs=False):
        #print("Start simulation")
        results=[]
        jobs=[stats for i in range(self.num_iter)]
        for j in jobs:
            results.append(self.monte_carlo(*j))
            #print(results)
        #print("done with current simulation")
        if probs:
            results=np.array(results)
            return len(results[results <= 0]) / len(results)
        else:
            return np.array(results)

    def simulate_ball(self,current_score,balls_remain,wickets_remain,rr,runs,runs_two,wb,er,sr,avg,ini=1,probs=False):
        if ini==1:
            results=self.simulate_first_half(stats=(current_score,balls_remain,wickets_remain,rr,runs,runs_two,wb,er,sr,avg))
            std=np.std(results)
            expected_score=np.mean(results)
        else:
            results=self.simulate_second_half(stats=(current_score,balls_remain,wickets_remain,rr,runs,runs_two,wb,er,sr,avg))
            std=np.std(results)
            expected_score=np.mean(results)
        if probs:
            return len(results[results <= 0]) / len(results)
        else:
            return results,expected_score,std

if __name__ == "__main__":
 print("main test method")
