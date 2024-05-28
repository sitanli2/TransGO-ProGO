#!/usr/bin/env python
# coding: utf-8

from pymongo import MongoClient
from gridfs import GridFS
import pandas as pd
import tqdm
import requests
import urllib.request as request #urllib3版本要<=1.25否则会与requests版本冲突
from bs4 import BeautifulSoup
from progressbar import ProgressBar
import json

# 建立mongodb连接
mongoClient = MongoClient('mongodb://localhost:27017/')
# mongoClient = MongoClient('mongodb://172.10.10.9:27019/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
# dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
dev = dtmr_dev.get_collection('TargetInfo')

# 创建GridFS对象，传入参数一：数据库，参数二：集合
gfs_af = GridFS(dtmr_dev, collection='AlphafoldModel_files')
gfs_fasta = GridFS(dtmr_dev, collection='FASTA_files')

#UPIDs = pd.read_excel("target_ID.xlsx").sort_values(by="No.").set_index("No.").Human_UPID.values

UPIDs =['A0jNW5','P05067','A0jP26','A1X283','A2A2Y4']
#panda.read_excel读取表格数据
'''
pandas中的sort_values()函数原理类似于SQL中的order by，可以将数据集依照某个字段中的数据进行排序，该函数即可根据指定列数据也可根据指定行的数据排序。
'''
records = []#创建空列表
no_topology = {}#创建空字典，其中topology为：蛋白结构域

No = 0
pbar = ProgressBar()
for UPID in  tqdm.tqdm(UPIDs):
    html_url = "https://rest.uniprot.org/uniprotkb/A1A519"#uniprot数据存放不同于pdb，Uniprot可直接获得数据包，不需要再用beautifulsoup对html源码进行解析。
    # https: // rest.uniprot.org / uniprotkb / P05067
    #html_url = "https://www.uniprot.org/uniprotkb/" + UPID
    response = request.urlopen(html_url)
    html = response.read()
    print('html:',html)

    '''此时不需要用beautifulsoup解析了直接对html操作就行'''
    print('UPID:',UPID)
    soup = BeautifulSoup(html, 'html.parser')#解析器为html.parser

    record = {}#空字典record
    No += 1
    record["No"] = No
    record["UPID"] = UPID

    overview = soup.find('div', id="entry-overview")#解析html源码，找到<div....,id="entry-overview"
    if overview == None:
        print('id:entry-overview抓取失败,没有这个字段')
        break
    record["Protein"] = overview.find("div", id="content-protein").text#find().text指截取text内容
    record["Gene"] = overview.find("div", id="content-gene").text
    record["Organism"] = overview.find("div", id="content-organism").text#organism：生物

    function_list = soup.find("div", id="function")#找到tag为<div...的第一个id="fuction"的字段
    function = {}#空字典function
    if function_list == None:
        print('id="function"抓取失败，没有这个字段')
        break
    for ul in function_list.find_all("ul", class_='noNumbering'):
        if ul.attrs != {}:
            name = ul.attrs["class"][1]
            function[name] = []
            for li in ul.find_all("li"):
                if li.attrs != {}:
                    pair = {}
                    pair["GO"] = li.attrs["class"][0]
                    pair["text"] = li.find("a").text
                        #                     print(link, text)
                    function[name].append(pair)

    record["Function"] = function#嵌套字典，即在record中在插入一个keyword:Fuction，内容为上述fuction{}的一个字典
    '''no_topology = {}#创建空字典'''
    topology_table = soup.find("table", id="topology_section")
    if topology_table == None:
        record["Topology"] = None
        no_topology[UPID] = record["Protein"]
    else:
        topology = []
        for tr in topology_table.find_all("tr"):
            row = {}
            for td in tr.find_all("td")[:3]:
                if td != []:
                    fkey = td.find("span", class_="context-help")
                    if fkey != None:
                        for s in fkey.select('span'):
                            s.extract()
                        if fkey != None:
                            row["Feature key"] = fkey.text

                    posi = td.find("a", class_="position tooltipped")
                    if posi != None:
                        posi.text
                        row["Positions"] = tuple(int(i) for i in posi.text.split("\xa0–\xa0"))

                    desp = td.find("span", property="text")
                    if desp != None:
                        row["Description"] = desp.text
            if row != {}:
                topology.append(row)

        record["Topology"] = topology#record列表创建新列表项topology：蛋白结构域

    r = requests.get('https://alphafold.ebi.ac.uk/files/AF-' + UPID + '-F1-model_v1.pdb', allow_redirects=True)
    if r.status_code != 200:
        record["AlphafoldModel"] = None
    else:
        record["AlphafoldModel"] = gfs_af.put(r.content, filename=UPID+".pdb", type="pdb", keyword=UPID)

    r = requests.get("https://www.uniprot.org/uniprot/" + UPID + ".fasta", allow_redirects=True)
    if r.status_code != 200:
        record["FASTA"] = None
    else:
        record["FASTA"] = gfs_fasta.put(r.content, filename=UPID+".fasta", type="fasta", keyword=UPID)

    records.append(record)

result = dev.insert_many(records)#mongodb insert_many
print(len(result.inserted_ids), 'records inserted!')

# with open('UniProt_data.json', 'w', encoding="utf-8") as fp:
#     json.dump(records, fp, ensure_ascii=False)


# with open('UniProt_no_topology.json', 'w', encoding="utf-8") as fp:
#     json.dump(no_topology, fp, ensure_ascii=False)

