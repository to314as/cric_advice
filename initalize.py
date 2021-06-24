import data_preprocessing
import train_models
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please enter the file path to data source with the -d option. Load a preprocesed dataframe with the -r option.')
    parser.add_argument('-d', type=str, default="./data/allT20/", help='file path to data directory')
    parser.add_argument('-o', type=int, default=20, help='give the number of overs in a game')
    parser.add_argument('-w', type=int, default=10, help='give the number of max wickets in a game')
    parser.add_argument('-r', dest='df', action='store_true', help='reload a preprocessed dataframe as data source')
    parser.add_argument('-s', type=str, default="./data/allT20/", help='file path to player stats data directory')
    parser.set_defaults(df=False)
    args=parser.parse_args()
    data_preprocessing.main(args)
    train_models.main(args)