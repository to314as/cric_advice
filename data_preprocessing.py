#imports
import pandas as pd
#import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import glob
import argparse
import datetime, time
import seaborn as sns
from datetime import datetime
import math
import os
import random
from tqdm import tqdm
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from numpy import nan
from collections import defaultdict
import statsmodels as sm
import sys
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Probit

parser = argparse.ArgumentParser(description='Please enter the file path to data source with the -d option. Load a preprocesed dataframe with the -r option.')
parser.add_argument('-d', type=str, default="./data/allT20/", help='file path to data directory')
parser.add_argument('-s', type=str, default="./data/alT20/", help='file path to player stats data directory')
parser.add_argument('-o', type=int, default=20, help='give the number of overs in a game')
parser.add_argument('-w', type=int, default=10, help='give the number of max wickets in a game')
parser.add_argument('-r', dest='df', action='store_true', help='reload a preprocessed dataframe as data source')
parser.set_defaults(df=False)
args=parser.parse_args()

over_segments_odi=[0,15,35,40,45,50]
wicket_segments_odi=[[0,0,0,0],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]]
over_segments_t20=[0,3,6,11,15,18,20]
wicket_segments_t20=[[0,0,0,0],[0,2,4,10],[0,3,5,10],[0,4,6,10],[0,4,6,10],[0,5,7,10],[0,6,8,10]]
metrics=['balls', 'games_striker', 'runs', 'dismissed', '4s', '6s', '50s', '100s', 's_r', 'Av', 'games_bowler', 'balls_bowl', 'concived_runs', 'wickets', 'noballs', 'wides', 'e_r','extras_bowler','extras_striker']

if args.o==50:
    over_segments=over_segments_odi
    wicket_segments=wicket_segments_odi
else:
    over_segments=over_segments_t20
    wicket_segments=wicket_segments_t20

def get_game_stats(df,ini=1,target=150):
    s=[[],[],[],[],[],[],[]]
    b=args.o*6
    w=args.w
    runs=0
    extras=0
    rr=0
    rrr=target/b
    player_runs=defaultdict(lambda: 0)
    for r in df.iterrows():
        i=r[0]
        r=r[1]
        s[0].append(b)
        s[1].append(w)
        s[2].append(rr)
        s[3].append(player_runs[r.striker])
        s[4].append(player_runs[r.non_striker])
        s[5].append(rrr)
        s[6].append(runs+extras)
        try:
            if not math.isnan(r.wicket_type):
                w-=1
            elif not (r.wides>0 or r.noballs>0):
                b-=1
                if (r.legbyes>0 or r.byes>0):
                    extras+=r.extras
                    #player_runs[r.striker]=player_runs[r.striker]+r.extras
                else:
                    runs+=r.runs_off_bat
                    player_runs[r.striker]=player_runs[r.striker]+r.runs_off_bat
            else:
                extras+=r.extras
            rr=(rr*(args.o*6-b)/1+r.runs_off_bat)*(1/(args.o*6-b+1))
        except:
            w-=1
            if not (r.wides>0 or r.noballs>0):
                b-=1
            extras+=r.extras
            rr=(rr*(args.o*6-b)/1+r.runs_off_bat)*(1/(args.o*6-b+1))
        if ini==1:
            rrr=0
        else:
            if b!=0:
                rrr=(target-(runs+extras))/b
            else:
                rrr=100
    return np.array(s)

#read in match data
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_stats(df):
    striker=pd.DataFrame()
    #batter stats
    s=df.groupby(['striker'])
    s_match=df.groupby(['striker','match_id'])
    striker["name"]=s['ball'].count().reset_index().striker
    striker["balls"]=s['ball'].count().reset_index().ball
    striker["games_striker"]=s['match_id'].nunique().reset_index().match_id
    striker["runs"]=s['runs_off_bat'].sum().reset_index().runs_off_bat
    striker["dismissed"]=s['wicket_scored'].sum().reset_index().wicket_scored
    striker["4s"]=s['runs_off_bat'].agg(lambda x: (x==4).sum()).reset_index().runs_off_bat/striker["balls"]
    striker["6s"]=s['runs_off_bat'].agg(lambda x: (x==6).sum()).reset_index().runs_off_bat/striker["balls"]
    striker["50s"]=(s_match['runs_off_bat'].sum()>50).groupby(['striker']).sum().reset_index().runs_off_bat
    striker["100s"]=(s_match['runs_off_bat'].sum()>100).groupby(['striker']).sum().reset_index().runs_off_bat
    striker["extras_striker"]=s['extras'].sum().reset_index().extras
    #bowler stats
    bowler=pd.DataFrame()
    b=df.groupby(['bowler'])
    bowler["name"]=b['ball'].count().reset_index().bowler
    bowler["games_bowler"]=s['match_id'].nunique().reset_index().match_id
    bowler["balls_bowl"]=b['ball'].count().reset_index().ball
    bowler["concived_runs"]=b['runs_off_bat'].sum().reset_index().runs_off_bat
    bowler["wickets"]=b['wicket_scored'].sum().reset_index().wicket_scored
    bowler["noballs"]=b['noballs'].sum().reset_index().noballs
    bowler["wides"]=b['wides'].sum().reset_index().wides
    bowler["extras_bowler"]=b['extras'].sum().reset_index().extras
    #central metrics
    striker["s_r"]=striker["runs"]/striker["balls"]*100
    striker["Av"]=striker["runs"]/striker["dismissed"]
    striker["Av"][striker["Av"]==np.inf]=striker["Av"][striker["Av"]==np.inf]=np.nan
    striker["Av"]=striker["Av"].fillna(np.nanmean(striker["Av"]))
    striker["w_b"]=striker["balls"]/striker["dismissed"]
    striker["w_b"][striker["w_b"]==np.inf]=striker["w_b"][striker["w_b"]==np.inf]=np.nan
    striker["w_b"]=striker["w_b"].fillna(np.nanmean(striker["w_b"]))
    bowler["e_r"]=bowler["concived_runs"]/((bowler["balls_bowl"]-b["extras"].sum().reset_index().extras)/6)
    player=striker.merge(bowler,on="name",how="outer")
    #player=player.sort_values(by="name")
    player=player.set_index("name")
    player=player.T
    return player

def preprocess(game_table):
    game_table.start_date=pd.to_datetime(game_table.start_date, format='%Y-%m-%d')
    #game_table.player_dismissed=game_table.player_dismissed.notnull()
    game_table["wicket_scored"]=game_table.wicket_type.notnull()
    game_table["overs"] = game_table.ball.astype(np.int16)
    game_table["new_no_ball"]=game_table.start_date>datetime(2015,5,7)
    game_table=game_table.sort_values(by=["start_date","innings","ball"])
    #game_table["w_b"]=game_table["Av_career"]/(game_table["s_r_career"]/100)
    #game_table=game_table[~game_table.match_id.isin(game_stat[-50:].match_id)]
    return game_table

def create_player_tables(args,players):
    stats=defaultdict()
    for m in metrics:
        stats[m]=pd.DataFrame()
        stats[m]["name"]=players
    all_files = sorted(glob.glob(args.d + "/*.csv"), key=numericalSort)
    print("compute players individual statistics over time")
    for filename in tqdm(all_files):
        #print(i,len(all_files))
        with open(filename, 'r') as f:
            df = preprocess(pd.read_csv(f))
            df_1=df[df.innings==1]
            #if len(df_1)<args.o*6:
            #    continue
            df_2=df[df.innings==2]
            m_id=df_1.match_id[0]
            date=df_1.start_date[0]
            team_1=df_1.batting_team[0]
            team_2=df_1.bowling_team[0]
            score_1=df_1.runs_off_bat.sum()+df_1.extras.sum()
            score_2=df_2.runs_off_bat.sum()+df_2.extras.sum()
            wickets_1=df_1.player_dismissed.notnull().sum()
            wickets_2=df_2.player_dismissed.notnull().sum()
            overs_1=len(df_1)-df_1["noballs"].sum()-df_1["wides"].sum()
            overs_2=len(df_2)-df_2["noballs"].sum()-df_2["wides"].sum()
            #if score_2<score_1 and wickets_2<args.w and args.o*6-overs_2>0:
            #    continue
            game_stats=get_stats(df)
            #make a pndas df for each metric
            temp_stats=defaultdict()
            for m in list(game_stats.index):
                temp_stats[m]=[]
            for i in players:
                if i in game_stats.columns:
                    for m in list(stats.keys()):
                        temp_stats[m].append(game_stats[i][m])
                else:
                    for m in list(stats.keys()):
                        temp_stats[m].append(np.nan)
            for m in list(stats.keys()):
                stats[m][df.start_date[0]]=temp_stats[m]
    for m in list(stats.keys()):
        stats[m]=stats[m].set_index("name")
        print(stats[m])
    return stats

def create_match_table(args):
    all_files = sorted(glob.glob(args.d + "*.csv"), key=numericalSort)
    print("Pre sorted match files")
    li = []
    i=0
    rain_int=0
    columns=["team_1","team_2","score_1","score_2","wickets_1","wickets_2","balls_played_1","balls_played_2","difference","match_id","start_date"]
    extra_columns=["balls_remain","wickets_remain","rr","sr","sr_two","rrr","current_score"]
    game_stat=pd.DataFrame(columns=columns)
    gs=[[],[],[],[],[],[],[],[],[],[],[]]
    print("start reading files...")
    for filename in tqdm(all_files):
        with open(filename, 'r') as f:
            df = pd.read_csv(f)
            #print(df)
            df=df[df.innings<3]
            df_1=df[df.innings==1]
            df["rain"] = [False for i in range(len(df))]
            if len(df_1)<args.o*6:
                #rain_int += 1
                #continue
                df["rain"]=[True for i in range(len(df))]
            df_2=df[df.innings==2]
            m_id=df_1.match_id[0]
            date=df_1.start_date[0]
            team_1=df_1.batting_team[0]
            team_2=df_1.bowling_team[0]
            score_1=df_1.runs_off_bat.sum()+df_1.extras.sum()
            score_2=df_2.runs_off_bat.sum()+df_2.extras.sum()
            wickets_1=df_1.player_dismissed.notnull().sum()
            wickets_2=df_2.player_dismissed.notnull().sum()
            overs_1=len(df_1)-df_1["noballs"].sum()-df_1["wides"].sum()
            overs_2=len(df_2)-df_2["noballs"].sum()-df_2["wides"].sum()
            if score_2<score_1 and wickets_2<args.w and overs_1-overs_2>0:
                rain_int+=1
                #continue
                df["rain"] = [True for i in range(len(df))]
            #curren_striker_stats=defaultdict()
            #curren_bowler_stats=defaultdict()
            #strikers=np.unique(df.striker)
            #bowlers=np.unique(df.bowler)
            #for m in metrics:
                #curren_striker_stats[m]=player_stats[m][strikers].iloc[:i].sum()
                #curren_bowler_stats[m]=player_stats[m][bowlers].iloc[:i].sum()
            #curren_bowler_stats["e_r_career"]=curren_bowler_stats["concived_runs"]/((curren_bowler_stats["balls_bowl"]-curren_bowler_stats["extras_bowler"])/6)
            #curren_striker_stats["Av_career"]=curren_striker_stats["runs"]/curren_striker_stats["dismissed"]
            #curren_striker_stats["s_r_career"]=curren_striker_stats["runs"]/curren_striker_stats["balls"]*100
            #print("SR:",curren_striker_stats["s_r_career"])
            stats1=get_game_stats(df_1,ini=1)
            stats2=get_game_stats(df_2,ini=2,target=score_1)
            stats=np.concatenate((stats1,stats2),axis=1)
            for c in range(len(extra_columns)):
                df[extra_columns[c]]=stats[c]
            #df["Av_career"]=[curren_striker_stats["Av_career"][player] if curren_striker_stats["Av_career"][player]!=0 else player_mean["Av"] for player in df.striker]
            #df["s_r_career"]=[curren_striker_stats["s_r_career"][player] if curren_striker_stats["s_r_career"][player]!=0  else player_mean["s_r"] for player in df.striker]
            #df["e_r_career"]=[curren_bowler_stats["e_r_career"][player] if curren_bowler_stats["e_r_career"][player]!=0 else player_mean["e_r"] for player in df.bowler]
            #df=preprocess_game(df)
            gs[0].append(team_1)
            gs[1].append(team_2)
            gs[2].append(score_1)
            gs[3].append(score_2)
            gs[4].append(wickets_1)
            gs[5].append(wickets_2)
            gs[6].append(overs_1)
            gs[7].append(overs_2)
            gs[8].append(score_1-score_2)
            gs[9].append(m_id)
            gs[10].append(date)
            li.append(df)
        i+=1
    print("Data loaded")
    print(rain_int,"matches with irregularities excluded")
    for c in range(len(columns)):
        game_stat[columns[c]]=gs[c]
    if not os.path.exists(args.d+"stats/"):
        os.makedirs(args.d+"stats/")
    game_stat.to_csv(args.d+"stats/"+"game_stats.csv")
    game_table = pd.concat(li, axis=0, ignore_index=True)
    return game_table,game_stat

def get_players_from_gametable(game_table):
    players=np.unique(game_table[["striker","non_striker","bowler"]])
    return players

def add_player_stats(game_table,player_stats):
    print("Add player stats to game_table...")
    selected_metric=["e_r_career","Av_career","s_r_career","extras_striker","extras_bowler","games_striker","games_bowler"]
    for sem in selected_metric:
        game_table[sem]=[nan for j in range(len(game_table))]
    i = 0
    for g_id in tqdm(np.unique(game_table.match_id)):
        df=game_table[game_table.match_id==g_id]
        curren_striker_stats=defaultdict()
        curren_bowler_stats=defaultdict()
        strikers=np.unique(df.striker)
        bowlers=np.unique(df.bowler)
        for m in metrics:
            curren_striker_stats[m]=player_stats[m][strikers].iloc[:i].sum()
            curren_bowler_stats[m]=player_stats[m][bowlers].iloc[:i].sum()
        curren_bowler_stats["e_r_career"]=curren_bowler_stats["concived_runs"]/((curren_bowler_stats["balls_bowl"]-curren_bowler_stats["extras_bowler"])/6)
        curren_bowler_stats["e_r_career"][curren_bowler_stats["e_r_career"]==np.inf]=curren_bowler_stats["e_r_career"][curren_bowler_stats["e_r_career"]==np.inf]=np.nan
        curren_bowler_stats["e_r_career"]=curren_bowler_stats["e_r_career"].fillna(np.nanmean(curren_bowler_stats["e_r_career"]))
        curren_striker_stats["Av_career"]=curren_striker_stats["runs"]/curren_striker_stats["dismissed"]
        curren_striker_stats["s_r_career"]=curren_striker_stats["runs"]/curren_striker_stats["balls"]*100
        for p in curren_striker_stats["balls"].keys():
            if curren_striker_stats["balls"][p]!=0:
                curren_striker_stats["extras_striker"][p] = curren_striker_stats["extras_striker"][p]/curren_striker_stats["balls"][p]
            else:
                curren_striker_stats["extras_striker"][p] = curren_striker_stats["extras_striker"][p]
        for p in curren_bowler_stats["balls_bowl"].keys():
            if curren_bowler_stats["balls_bowl"][p]!=0:
                curren_bowler_stats["extras_bowler"][p] = curren_bowler_stats["extras_bowler"][p] / curren_bowler_stats["balls_bowl"][p]
            else:
                curren_bowler_stats["extras_bowler"][p] = curren_bowler_stats["extras_bowler"][p]
        #print(curren_striker_stats)
        game_table["Av_career"].loc[game_table.match_id==g_id]=[curren_striker_stats["Av_career"][player] if curren_striker_stats["dismissed"][player]!=0 else np.nanmean(curren_striker_stats["Av_career"]) for player in df.striker]
        game_table["s_r_career"].loc[game_table.match_id==g_id]=[curren_striker_stats["s_r_career"][player] if curren_striker_stats["balls"][player]!=0  else np.nanmean(curren_striker_stats["s_r_career"]) for player in df.striker]
        game_table["e_r_career"].loc[game_table.match_id==g_id]=[curren_bowler_stats["e_r_career"][player] if curren_bowler_stats["balls_bowl"][player]!=0 else np.nanmean(curren_bowler_stats["e_r_career"])  for player in df.bowler]
        game_table["extras_bowler"].loc[game_table.match_id == g_id] = [curren_bowler_stats["extras_bowler"][player] for player in df.bowler]
        game_table["extras_striker"].loc[game_table.match_id == g_id] = [curren_striker_stats["extras_striker"][player] for player in df.striker]
        game_table["games_bowler"].loc[game_table.match_id == g_id] = [curren_bowler_stats["games_bowler"][player] for player in df.bowler]
        game_table["games_striker"].loc[game_table.match_id == g_id] = [curren_striker_stats["games_striker"][player] for player in df.striker]
        i+=1
    return game_table

def preprocess_gametable(game_table):
    game_table.start_date=pd.to_datetime(game_table.start_date, format='%Y-%m-%d')
    #game_table.player_dismissed=game_table.player_dismissed.notnull()
    game_table["wicket_scored"]=game_table.wicket_type.notnull()
    game_table["overs"] = game_table.ball.astype(np.int8)
    game_table["new_no_ball"]=game_table.start_date>datetime(2015,5,7)
    game_table=game_table.sort_values(by=["start_date","innings","ball"])
    for c in ["Av_career","s_r_career","e_r_career"]:
        game_table[c][game_table[c]==np.inf]=game_table[c][game_table[c]==np.inf]=np.nan
        game_table[c]=game_table[c].fillna(np.nanmean(game_table[c]))
    game_table["w_b"]=game_table["Av_career"]/(game_table["s_r_career"]/100)
    game_table["w_b"][game_table["w_b"]==np.inf]=game_table["w_b"][game_table["w_b"]==np.inf]=np.nan
    game_table["w_b"]=game_table["w_b"].fillna(np.nanmean(game_table["w_b"]))
    game_table["striker_performance"] = game_table["w_b"]*game_table["s_r_career"]
    #game_table=game_table[~game_table.match_id.isin(game_stat[-50:].match_id)]
    return game_table

def main(args):
    print("Start processing data from folder "+args.d)
    game_table,game_stat=create_match_table(args)
    players=get_players_from_gametable(game_table)
    player_stats=create_player_tables(args,players)
    for m in list(player_stats.keys()):
        player_stats[m]=player_stats[m].transpose()
        player_stats[m].to_csv(args.d+"stats/"+str(m)+"_player_stats.csv",index_label="date")
    for m in list(player_stats.keys()):
        player_stats[m]=pd.read_csv(args.s+"stats/"+str(m)+"_player_stats.csv")
    game_table=add_player_stats(game_table,player_stats)
    game_table=preprocess_gametable(game_table)
    print("Save game_table")
    game_table.to_csv(args.d+"stats/game_table.csv")
    norm_game_table = game_table.copy()
    for m in ["balls_remain", "wickets_remain", "rr", "rrr", "sr", "w_b", "e_r_career", "Av_career", "s_r_career", "extras_striker", "extras_bowler"]:
        norm_game_table[m] = game_table[m]
        norm_game_table[m] = (game_table[m] - game_table[m].min()) / (game_table[m].max() - game_table[m].min())
    norm_game_table.to_csv(args.d + "stats/norm_game_table.csv")


if __name__ == "__main__":
    game_table,game_stat=create_match_table(args)
    players=get_players_from_gametable(game_table)
    player_stats=create_player_tables(args,players)
    for m in list(player_stats.keys()):
        player_stats[m]=player_stats[m].transpose()
        player_stats[m].to_csv(args.d+"stats/"+str(m)+"_player_stats.csv",index_label="date")
    game_table=add_player_stats(game_table,player_stats)
    game_table=preprocess_gametable(game_table)
    print("Save game_table")
    game_table.to_csv(args.d + "stats/game_table.csv")
    norm_game_table=game_table.copy()
    for m in ["balls_remain","wickets_remain","rr","rrr","sr","w_b","e_r_career","Av_career","s_r_career"]:
        norm_game_table[m]=game_table[m]
        norm_game_table[m]=(game_table[m] - game_table[m].min()) / (game_table[m].max() - game_table[m].min())
    norm_game_table.to_csv(args.d + "stats/norm_game_table.csv")