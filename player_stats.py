import numpy as np
import pandas as pd
from collections import defaultdict

class PlayerStats:

    def __init__(self,d="./data/allT20/"):
        self.d=d
        self.game_table=pd.read_csv(self.d+"stats/game_table.csv")
        self.player_stats=defaultdict()
        self.metrics=['balls', 'games_striker', 'runs', 'dismissed', '4s', '6s', '50s', '100s', 's_r', 'Av', 'games_bowler', 'balls_bowl', 'concived_runs', 'wickets', 'noballs', 'wides', 'e_r','extras_bowler','extras_striker']
        for m in self.metrics:
            self.player_stats[m]=pd.read_csv(self.d+"stats/"+str(m)+"_player_stats.csv")

    def get_players_of_country(self,country):
        strikers=np.unique(self.game_table[self.game_table.batting_team==country].striker)
        bowlers=np.unique(self.game_table[self.game_table.bowling_team==country].bowler)
        return list(strikers),list(bowlers)

    def get_best_strikers(self,country):
        strikers,bowlers=self.get_players_of_country(country)
        ps=self.player_stats[strikers].T
        ps=ps[ps.games_striker>1][ps.balls>10][ps.w_b>=10].sort_values(by="s_r",ascending=False).iloc[:11]
        return ps

    def get_best_bowlers(self,country):
        strikers,bowlers=self.get_players_of_country(country)
        pb=self.player_stats[bowlers].T
        pb=pb[pb.games_bowler>1][pb.balls_bowl>10].sort_values(by="e_r").iloc[:11]
        return pb

    def get_metric_players(self,players=[]):
        return self.player_stats[players]

    def get_required_metric_players(self,strikers=[],bowlers=[],date=pd.to_datetime('20210101', format='%Y%m%d')):
        curren_striker_stats=defaultdict()
        curren_bowler_stats=defaultdict()
        s=[]
        b=[]
        for i in strikers:
            if i in self.player_stats[self.metrics[0]]:
                s.append(i)
            else:
                for j in strikers:
                    if j in self.player_stats[self.metrics[0]]:
                        for m in self.metrics:
                            self.player_stats[m][i] = self.player_stats[m][j]
                        s.append(i)
                        break
                print(i)
        for i in bowlers:
            if i in self.player_stats[self.metrics[0]]:
                b.append(i)
            else:
                for j in bowlers:
                    if j in self.player_stats[self.metrics[0]]:
                        for m in self.metrics:
                            self.player_stats[m][i]=self.player_stats[m][j]
                        b.append(i)
                        break
                print(i)
        for m in self.metrics:
            curren_striker_stats[m]=self.player_stats[m][s].loc[pd.to_datetime(self.player_stats[m].date)<date].sum()
            curren_bowler_stats[m]=self.player_stats[m][b].loc[pd.to_datetime(self.player_stats[m].date)<date].sum()
        curren_bowler_stats["e_r_career"]=curren_bowler_stats["concived_runs"]/((curren_bowler_stats["balls_bowl"]-curren_bowler_stats["extras_bowler"])/6)
        curren_bowler_stats["e_r_career"][curren_bowler_stats["e_r_career"]==np.inf]=curren_bowler_stats["e_r_career"][curren_bowler_stats["e_r_career"]==np.inf]=np.nan
        curren_bowler_stats["e_r_career"]=curren_bowler_stats["e_r_career"].fillna(np.max(curren_bowler_stats["e_r_career"]))
        curren_striker_stats["Av_career"]=curren_striker_stats["runs"]/curren_striker_stats["dismissed"]
        curren_striker_stats["s_r_career"]=curren_striker_stats["runs"]/curren_striker_stats["balls"]*100
        curren_striker_stats["Av_career"][curren_striker_stats["Av_career"]==np.inf]=curren_striker_stats["Av_career"][curren_striker_stats["Av_career"]==np.inf]=np.nan
        curren_striker_stats["Av_career"]=curren_striker_stats["Av_career"].fillna(np.nanmean(curren_striker_stats["Av_career"]))
        curren_striker_stats["s_r_career"][curren_striker_stats["s_r_career"]==np.inf]=curren_striker_stats["s_r_career"][curren_striker_stats["s_r_career"]==np.inf]=np.nan
        curren_striker_stats["s_r_career"]=curren_striker_stats["s_r_career"].fillna(np.nanmean(curren_striker_stats["s_r_career"]))
        curren_striker_stats["w_b_career"]=curren_striker_stats["Av_career"]/(curren_striker_stats["s_r_career"]/100)
        curren_striker_stats["w_b_career"][curren_striker_stats["w_b_career"] == np.inf] = \
        curren_striker_stats["w_b_career"][curren_striker_stats["w_b_career"] == np.inf] = np.nan
        curren_striker_stats["w_b_career"] = curren_striker_stats["w_b_career"].fillna(
            np.nanmean(curren_striker_stats["w_b_career"]))
        for p in curren_striker_stats["games_striker"].keys():
            if curren_striker_stats["games_striker"][p]!=0:
                curren_striker_stats["extras_striker"][p] = curren_striker_stats["extras_striker"][p] / curren_striker_stats["games_striker"][p]
            else:
                curren_striker_stats["extras_striker"][p] = 0
        for p in curren_bowler_stats["games_bowler"].keys():
            if curren_bowler_stats["games_bowler"][p]!=0:
                curren_bowler_stats["extras_bowler"][p] = curren_bowler_stats["extras_bowler"][p] / curren_bowler_stats["games_bowler"][p]
            else:
                curren_bowler_stats["extras_bowler"][p] = 0
        return curren_bowler_stats, curren_striker_stats

if __name__ == "__main__":
    player_stats=PlayerStats()
    player_stats.get_required_metric_players()
