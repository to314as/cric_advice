from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import os
import shutil
from bs4 import BeautifulSoup
import requests

#"https://cricsheet.org/downloads/it20s_male_csv2.zip"
links=["https://cricsheet.org/downloads/t20s_male_csv2.zip","https://cricsheet.org/downloads/bbl_male_csv2.zip","https://cricsheet.org/downloads/ipl_male_csv2.zip","https://cricsheet.org/downloads/ntb_male_csv2.zip","https://cricsheet.org/downloads/psl_male_csv2.zip","https://cricsheet.org/downloads/cpl_male_csv2.zip"]
dir=["T20I","BigBash","IPL","T20Blast","PSL","CPL"]
for zipurl,d in zip(links,dir):
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall('./data/'+d)

dir=["T20I","BigBash","IPL","T20Blast","PSL","CPL"]
for d in dir:
    print("Downloading data:",d)
    src_dir='./data/'+d
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.csv'):
                try:
                    shutil.move(os.path.join(root,f), '.\\data\\allT20')
                except:
                    continue