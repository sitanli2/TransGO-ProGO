from pymongo import MongoClient
from gridfs import GridFS
import psycopg2

import pandas as pd

import tqdm
import requests
import urllib.request as request
from bs4 import BeautifulSoup
from progressbar import ProgressBar
import json

# 建立mongodb连接
# mongoClient = MongoClient('mongodb://localhost:27017/')
mongoClient = MongoClient('mongodb://172.10.10.48:27017/') 
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
dev = dtmr_dev.get_collection('TargetInfo')
gfs_pdb = GridFS(dtmr_dev, collection='PDB_files')


UPIDs = pd.read_excel("target_ID.xlsx").sort_values(by="No.").set_index("No.").Human_UPID.values

PDB_IDS_FOR_FILES = set()

for UPID in tqdm.tqdm(UPIDs):

    html_url = "https://www.uniprot.org/uniprot/" + UPID
    response = request.urlopen(html_url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    
    cross_references = soup.find("div", id="cross_references")
    structureTable = cross_references.find("table", class_="databaseTable STRUCTURE")
    found_PDB = False
    for tr in structureTable.find_all("tr"):
        tds = tr.find_all("td")
        if 'PDB entry' in tds[1].text:
            next_tr = tr.find_next_siblings("tr")
            found_PDB = True
            break
    
    if found_PDB:
        pdb_ids = []
        for tr in next_tr:
            for a in tr.find_all("a", class_="pdb"):
                pdb_ids.append(a.text)
        PDB_IDS_FOR_FILES.update(pdb_ids)
    else:
        pdb_ids = None

    
    query = { "UPID": UPID }
    newvalues = { "$set": { "PDB": pdb_ids } }

    dev.update_one(query, newvalues)

pd.DataFrame(list(PDB_IDS_FOR_FILES), columns=["pdb_id"]).to_csv("PDB_IDS_FOR_FILES.csv", index=False)

noFile = []
for pdb_id in tqdm.tqdm(list(PDB_IDS_FOR_FILES)):
    r = requests.get("https://files.rcsb.org/download/" + pdb_id + ".pdb", allow_redirects=True)
    if r.status_code != 200:
        noFile.append(pdb_id)
    else:
        gfs_pdb.put(r.content, filename=pdb_id+".pdb", type="pdb", keyword=pdb_id)


pd.DataFrame(noFile, columns=["pdb_id"]).to_csv("pdb_id_file_missing.csv", index=False)



