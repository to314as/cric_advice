import pickle
from datetime import date,datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")
from segmented_monte_carlo import SegmentedMonteCarlo
from player_stats import PlayerStats
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
import time
from live_cricket import LiveCricket

class ModelTester:
    def __init__(self,d="./data/allT20/",l="./data/PSL/",o=20,date=datetime(2015, 5, 7)):
        self.o=o
        self.w=10
        self.d=d
        self.num_iter=1000
        self.num_games=5
        self.metrics=['balls', 'games_striker', 'runs', 'dismissed', '4s', '6s', '50s', '100s', 's_r', 'Av', 'games_bowler', 'balls_bowl', 'concived_runs', 'wickets', 'noballs', 'wides', 'e_r','extras_bowler','extras_striker']
        if self.o==50: #odi matches
            self.over_segments=np.array([0,15,35,40,45,50])
            self.wicket_segments=np.array([[0,0,0,0],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]])
        else: #t20 matches
            self.over_segments=np.array([0,6,11,14,17,20])
            self.wicket_segments=np.array([[0,0,0,0],[0,1,2,10],[0,2,4,10],[0,3,5,10],[0,4,6,10],[0,5,7,10],[0,5,7,10]])
        game_table = pd.read_csv(d + "stats/game_table.csv")
        game_table.start_date = pd.to_datetime(game_table.start_date, format='%Y-%m-%d')
        game_table = game_table[game_table.start_date > date]
        self.game_table=game_table
        game_stats = pd.read_csv(l + "stats/game_stats.csv")
        game_stats.start_date = pd.to_datetime(game_stats.start_date, format='%Y-%m-%d')
        self.game_stats = game_stats[game_stats.start_date > date]
        self.sizes=pickle.load(open(self.d + "sizes.p", "rb"))
        self.sizes2 = pickle.load(open(self.d + "sizes2.p", "rb"))
        self.os = pickle.load(open(self.d + "os.p", "rb"))
        self.ps = pickle.load(open(self.d + "ps.p", "rb"))
        self.es = pickle.load(open(self.d + "es.p", "rb"))
        self.ess = pickle.load(open(self.d + "ess.p", "rb"))
        self.os2 = pickle.load(open(self.d + "os2.p", "rb"))
        self.ps2 = pickle.load(open(self.d + "ps2.p", "rb"))
        self.es2 = pickle.load(open(self.d + "ps.p", "rb"))
        self.ess2 = pickle.load(open(self.d + "ess2.p", "rb"))

    def test_inning(self,ini,visual=True,half=1):
        target = self.game[self.game.innings == 1].runs_off_bat.sum() + self.game[
            self.game.innings == 1].extras.sum()
        print(target)
        df_segment = self.game[self.game.innings == half]
        score = []
        results = []
        jobs = []
        ball_number = []
        order = np.array(["" for i in range(11)], dtype=object)
        pool = mp.Pool(self.num_cores)
        for r in df_segment.iterrows():
            ball = r[1]
            pos = int(10 - ball.wickets_remain)
            order[pos] = ball.striker
            played = [i for i in order if i != ""]
            to_play = [i for i in self.strikers["s_r_career"].index if (i not in played)]
            ns = np.concatenate((played, to_play))
            try:
                self.players_sr = self.players_sr.reindex(ns)
                self.players_av = self.players_av.reindex(ns)
                self.players_wb = self.players_wb.reindex(ns)
                self.extras_strikers = self.extras_strikers.reindex(ns)
            except:
                self.players_sr = self.players_sr
            score.append(ball.current_score)
            ball_number.append(self.o * 6 - ball.balls_remain)
            if ini == 1:
                jobs.append((ball.current_score, ball.balls_remain, ball.wickets_remain, ball.rr, ball.sr, ball.sr_two,
                             self.players_wb, self.players_er, self.players_sr, self.players_av, self.extras_strikers,
                             self.extras_bowlers, ini, len(self.over_segments)-1))
            else:
                jobs.append((target-ball.current_score, ball.balls_remain, ball.wickets_remain, ball.rr, ball.sr, ball.sr_two,
                             self.players_wb, self.players_er, self.players_sr, self.players_av, self.extras_strikers,
                             self.extras_bowlers, ini, len(self.over_segments)-1))
        # print("Start parallel computation")
        t = time.time()
        with parallel_backend('loky', n_jobs=self.num_cores):
            results = Parallel()(delayed(self.monte_carlo.simulate_segment)(j) for j in jobs)
        # for j in jobs:
        # result=self.monte_carlo.simulate_segment(j)
        # print(result)
        # results.append(result)
        print("Time taken:", time.time() - t)
        results = np.array(results)
        # print(results)
        std = np.std(results, axis=1)
        projected_score = np.mean(results, axis=1)
        pool.close()
        if visual:
            if ini==1:
                plt.figure()
                plt.plot(ball_number,np.array(projected_score))
                plt.plot(ball_number, [target for i in range(len(projected_score))])
                plt.fill_between(ball_number, np.array(projected_score) - np.array(std),
                                 np.array(projected_score) + np.array(std), alpha=0.1)
                plt.plot(ball_number,score)
                plt.show()
            else:
                plt.figure()
                plt.plot(ball_number, target-np.array(projected_score))
                plt.plot(ball_number, [target for i in range(len(projected_score))])
                plt.fill_between(ball_number, target-np.array(projected_score) - np.array(std),
                                 target-np.array(projected_score) + np.array(std), alpha=0.1)
                plt.plot(ball_number, score)
                plt.show()
            if half==2:
                probs = []
                for r in results:
                    r = np.array(r)
                    probs.append(len(r[r>0]) / len(r))
                print(probs)
                probs = np.array(probs)
                probs=np.pad(probs, (0, max(0,len(self.file["1st"])-len(probs))), 'edge')
                #np.savetxt(self.file, probs,delimiter=',',newline=",",fmt="%1.2f")
                self.file["2nd"] = probs
                plt.figure()
                plt.plot(probs)
                plt.plot((1 - probs))
                plt.legend([self.t_o, self.t_t])
                plt.show()
        return results, std, score

    def single_test(self,ini=1,half=1):
        target = self.game[self.game.innings == 1].runs_off_bat.sum() + self.game[
            self.game.innings == 1].extras.sum()
        print(target)
        df_segment = self.game[self.game.innings == half]
        ball=df_segment.iloc[0]
        j=(ball.current_score, ball.balls_remain, ball.wickets_remain, ball.rr, ball.sr, ball.sr_two,
                     self.players_wb, self.players_er, self.players_sr, self.players_av, self.extras_strikers,
                     self.extras_bowlers, ini, len(self.over_segments) - 1)
        df_segment = self.game[self.game.innings == half]
        result=self.monte_carlo.simulate_segment(j)
        #print(result)
        return result


    def test_segment(self,ini=1,seg=1,visual=False,half=1):
        target=self.game[self.game.innings == 1].runs_off_bat.sum()+self.game[self.game.innings == 1].extras.sum()
        df_segment = self.game[self.game.innings == half]
        df_segment_ij = df_segment[self.over_segments[seg - 1] <= df_segment.ball][df_segment.ball < self.over_segments[seg]]
        if len(df_segment_ij)<1:
            return [[target]],[[0]],[[target]]
        print("Over segment:",self.over_segments[seg])
        score = []
        results=[]
        jobs = []
        ball_number=[]
        order = np.array(["" for i in range(11)], dtype=object)
        pool = mp.Pool(self.num_cores)
        for r in df_segment_ij.iterrows():
            ball = r[1]
            pos = int(10 - ball.wickets_remain)
            order[pos] = ball.striker
            played = [i for i in order if i != ""]
            to_play = [i for i in self.strikers["s_r_career"].index if (i not in played)]
            ns = np.concatenate((played, to_play))
            try:
                self.players_sr = self.players_sr.reindex(ns)
                self.players_av = self.players_av.reindex(ns)
                self.players_wb = self.players_wb.reindex(ns)
                self.extras_strikers = self.extras_strikers.reindex(ns)
            except:
                self.players_sr = self.players_sr
            score.append(ball.current_score)
            ball_number.append(self.o*6-ball.balls_remain)
            if ini==1:
                jobs.append((ball.current_score, ball.balls_remain, ball.wickets_remain, ball.rr, ball.sr, ball.sr_two,
                 self.players_wb, self.players_er, self.players_sr, self.players_av, self.extras_strikers, self.extras_bowlers,ini,seg))
            else:
                jobs.append((target-ball.current_score, ball.balls_remain, ball.wickets_remain, ball.rrr, ball.sr, ball.sr_two,
                             self.players_wb, self.players_er, self.players_sr, self.players_av, self.extras_strikers, self.extras_bowlers,ini,seg))
        #print("Start parallel computation")
        t = time.time()
        with parallel_backend('loky', n_jobs=self.num_cores):
            results = Parallel()(delayed(self.monte_carlo.simulate_segment)(j) for j in jobs)
        #for j in jobs:
            #result=self.monte_carlo.simulate_segment(j)
            #print(result)
            #results.append(result)
        print("Time taken:",time.time() - t)
        results = np.array(results)
        #print(results)
        std = np.std(results, axis=1)
        projected_score = np.mean(results, axis=1)
        pool.close()
        if visual:
            if ini==1:
                plt.figure()
                plt.plot(ball_number,np.array(projected_score))
                plt.plot(ball_number, [target for i in range(len(projected_score))])
                plt.fill_between(ball_number, np.array(projected_score) - np.array(std),
                                 np.array(projected_score) + np.array(std), alpha=0.1)
                plt.plot(ball_number,score)
                plt.show()
            else:
                plt.figure()
                plt.plot(ball_number, target-np.array(projected_score))
                plt.plot(ball_number, [target for i in range(len(projected_score))])
                plt.fill_between(ball_number, target-np.array(projected_score) - np.array(std),
                                 target-np.array(projected_score) + np.array(std), alpha=0.1)
                plt.plot(ball_number, score)
                plt.show()
        return results, std, score

    def set_players(self,s=None,b=None):
        self.bowlers, self.strikers = self.player_stats.get_required_metric_players(s, b, self.date)
        while len(self.bowlers) < 11:
            self.bowlers=self.bowlers.append(self.bowlers[int(np.random.rand() * len(self.bowlers))])
        while len(self.strikers) < 11:
            self.strikers=self.strikers.append(self.strikers[int(np.random.rand() * len(self.strikers))])
        self.players_er = self.bowlers["e_r_career"]
        self.extras_bowlers = self.bowlers["extras_bowler"]
        self.players_sr = self.strikers["s_r_career"]
        self.players_av = self.strikers["Av_career"]
        self.players_wb = self.strikers["w_b_career"]
        self.extras_strikers = self.strikers["extras_striker"]
        self.players_sr.sort_values(ascending=False, inplace=True)
        self.players_av = self.players_av.reindex(self.players_sr.index)
        self.players_wb = self.players_wb.reindex(self.players_sr.index)
        self.extras_strikers = self.extras_strikers.reindex(self.players_sr.index)
        self.players_er.sort_values(ascending=True, inplace=True)
        self.extras_bowlers = self.extras_bowlers.reindex(self.players_er.index)
        print(self.players_sr)
        print(self.extras_strikers)
        print(self.players_er)
        print(self.extras_bowlers)
        print(self.extras_strikers)

    def team_test(self,b,s,min=100,max=250):
        self.set_players(b, s)
        probs=[]
        results=self.single_test(ini=1, half=2)
        for t in range(min,max):
            p=len(results[results>t])/len(results)
            probs.append(p)
        return probs

    def test(self):
        self.monte_carlo = SegmentedMonteCarlo(d=self.d, num_iter=self.num_iter)
        game_table = self.game_table
        #match_samples=self.game_stats['match_id'].sample(n=self.num_games, random_state=13).values
        match_samples = [1243020]#self.game_stats['match_id'].values[-5:]
        segment_error=[[] for i in range(len(self.over_segments)-1)]
        segment_error_percent = [[] for i in range(len(self.over_segments)-1)]
        print(match_samples)
        for mid in match_samples:
            print(mid)
            actual_segment_score = []
            predicted_segment_score = []
            self.game = game_table[game_table.match_id == mid]
            self.date = pd.to_datetime(self.game.start_date.iloc[0])
            lc = LiveCricket(mid=mid)
            self.player_stats = PlayerStats(d="./data/allT20/")
            self.num_cores = mp.cpu_count()
            team = lc.get_team_data()
            t_o = team["team_general_name"][0]
            t_t = team["team_general_name"][1]
            if self.game.batting_team.values[0]==t_o:
                self.t_o, self.t_t=t_o,t_t
                print(t_o, t_t)
                s, b = lc.get_player_data()
                s = list(set(s + list(np.unqiue(self.game[self.game.innings == 1].striker))))
                b = list(set(b + list(np.unqiue(self.game[self.game.innings == 1].bowler))))
            else:
                self.t_o, self.t_t=t_t,t_o
                print(t_t,t_o)
                b, s = lc.get_player_data()
                s = list(set(s + list(np.unique(self.game[self.game.innings == 1].striker))))
                b = list(set(b + list(np.unique(self.game[self.game.innings == 1].bowler))))
            #lc.get_online_stats()
            s = list(np.unique(s))
            while len(s) < 11:
                s.append(s[int(np.random.rand() * len(s))])
            print(s)
            b = list(np.unique(b))
            while len(b) < 11:
                b.append(b[int(np.random.rand() * len(b))])
            print(b)
            self.set_players(s,b)
            for seg in range(1,len(self.over_segments)):
                results, std, score=self.test_segment(ini=1,seg=seg)
                projected_score = np.mean(results, axis=1)
                actual_segment_score.append((score[-1]))
                predicted_segment_score.append(projected_score)
                error=np.mean(predicted_segment_score[seg-1]-actual_segment_score[seg-1])
                print("Error:",error)
                segment_error[seg-1].append(error)
                segment_error_percent[seg-1].append(error/score[-1])
            print(segment_error)
            print(segment_error_percent)
            target=self.game[self.game.innings == 1].runs_off_bat.sum()+self.game[self.game.innings == 1].extras.sum()
            self.test_inning(ini=1)
            self.set_players(b, s)
            actual_segment_score = []
            predicted_segment_score = []
            for seg in range(1,len(self.over_segments)):
                results, std, score=self.test_segment(ini=1,seg=seg,visual=False,half=2)
                projected_score = np.mean(results, axis=1)
                actual_segment_score.append((score[-1]))
                predicted_segment_score.append(projected_score)
                error=np.mean((target-predicted_segment_score[seg-1])-actual_segment_score[seg-1])
                print("Error:",error)
                segment_error[seg-1].append(error)
                segment_error_percent[seg-1].append(error/score[-1])
            print(segment_error)
            print(segment_error_percent)
            self.test_inning(ini=1,half=2)

    def test_innings(self):
        self.monte_carlo = SegmentedMonteCarlo(d=self.d, num_iter=self.num_iter)
        game_table = self.game_table
        # match_samples=self.game_stats['match_id'].sample(n=self.num_games, random_state=13).values
        match_samples = self.game_stats['match_id'].values[-8:]
        segment_error = [[] for i in range(len(self.over_segments) - 1)]
        segment_error_percent = [[] for i in range(len(self.over_segments) - 1)]
        print(match_samples)
        for mid in match_samples:
            print(mid)
            actual_segment_score = []
            predicted_segment_score = []
            self.game = game_table[game_table.match_id == mid]
            self.date = pd.to_datetime(self.game.start_date.iloc[0])
            self.file_name = self.d + "match_records/" + str(mid) + "_" + str(self.date)[:10]+".csv"
            self.file = pd.DataFrame(columns=["1st","2nd"])
            lc = LiveCricket(mid=mid)
            self.player_stats = PlayerStats(d="./data/allT20/")
            self.num_cores = mp.cpu_count()
            team = lc.get_team_data()
            t_o = team["team_general_name"][0]
            t_t = team["team_general_name"][1]
            if self.game.batting_team.values[0] == t_o:
                self.t_o, self.t_t = t_o, t_t
                print(t_o, t_t)
                s, b = lc.get_player_data()
                s = list(set(s + list(np.unique(self.game[self.game.innings == 1].striker))))
                b = list(set(b + list(np.unique(self.game[self.game.innings == 1].bowler))))
            else:
                self.t_o, self.t_t = t_t, t_o
                print(t_t, t_o)
                b, s = lc.get_player_data()
                s = list(set(s + list(np.unique(self.game[self.game.innings == 1].striker))))
                b = list(set(b + list(np.unique(self.game[self.game.innings == 1].bowler))))
            # lc.get_online_stats()
            s = list(np.unique(s))
            while len(s) < 11:
                s.append(s[int(np.random.rand() * len(s))])
            print(s)
            b = list(np.unique(b))
            while len(b) < 11:
                b.append(b[int(np.random.rand() * len(b))])
            print(b)
            mint, maxt = 100, 250
            probs = np.array(self.team_test(b, s, mint, maxt))
            print(probs)
            self.set_players(s, b)
            results, std, score = self.test_inning(ini=1)
            projected_score = np.mean(results, axis=1)
            probs_one=[]
            for result in results:
                ind=np.array(result - mint).astype(np.int)
                ind[ind>maxt-mint-1]=maxt-mint-1
                ind[ind<0]=0
                probs_one.append(np.mean(probs[ind]))
            #probs_one = np.array(probs[np.array(projected_score - mint).astype(np.int)])
            print(probs_one)
            probs_one=np.array(probs_one)
            plt.figure()
            plt.plot(probs_one)
            plt.plot((1 - probs_one))
            plt.legend([self.t_o, self.t_t])
            plt.show()
            df1 = self.game[self.game.innings == 1]
            df2 = self.game[self.game.innings == 2]
            if len(df1)<len(df2):
                probs_one = np.pad(probs_one, (0, max(0, len(df2) - len(probs_one))), 'edge')
            self.file["1st"]=probs_one
            self.set_players(b, s)
            self.test_inning(ini=2, half=2)
            self.file.to_csv(self.file_name)

        return

if __name__ == "__main__":
    model_trainer=ModelTester()
    model_trainer.test_innings()