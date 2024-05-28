from pymongo import MongoClient
from bson import ObjectId
import tqdm
#tqdm库用来生成进度条
import urllib.request as request #urllib3版本要<=1.25否则会与requests版本冲突
#urllin库用于操作网页 URL，并对网页的内容进行抓取处理。
from progressbar import ProgressBar
from jsonpath import jsonpath
from tkinter import _flatten #将二维列表转成一维列表
import requests
from bs4 import BeautifulSoup
import copy
import time #记录时间存入log

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')#数据库dtmr_dev的密码也是dtmr_dev
dev2 = dtmr_dev.get_collection('chembl_Target_test')

target_list = []
target_id_list = ['CHEMBL2074']
count = 0

for i in target_id_list:
    print('此时的target id=', i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/target/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')

    target_dict = {}
    # print(all_target[0].target_chembl_id.string)  #可见target_chembl_id不在cross_references下面 我可以通过这个来获取target_chembl_id ！！！
    target_dict['target_chembl_id'] = soup.target.target_chembl_id.string
    target_dict['NAME'] = soup.target.pref_name.string
    #print('accession error find:',bool(soup.target.target_components.target_component))
    if (bool(soup.target.target_components.target_component) != False):
        target_dict['UniProt_Scrapy Accessions'] = soup.target.target_components.target_component.accession.string
    else:
        target_dict['UniProt_Scrapy Accessions'] = 'null'
    if (bool(soup.target.target_components.target_component) != False):
        target_dict['Type'] = soup.target.target_components.target_component.relationship.string
    else:
        target_dict['Type'] = 'null'
    target_dict['Organism'] = soup.target.organism.string
    target_component_synonyms = {}
    component_synonym = []
    syn_type = []
    # target_component_xrefs = {}
    # xref_id = []
    # xref_name = []
    # xref_src_db = []
    all_synonums = soup.target.target_components.target_component.target_component_synonyms.find_all('target_component_synonym',recursive=False)
    all_xrefs = soup.target.target_components.target_component.target_component_xrefs.find_all('target',recursive=False)

    i=0
    while (i <= len(all_synonums)-1):
        component_synonym.append(all_synonums[i].component_synonym.string)
        syn_type.append(all_synonums[i].syn_type.string)
        i += 1
    target_component_synonyms['component_synonym'] = component_synonym
    target_component_synonyms['syn_type'] = syn_type
    target_dict['target_component_synonyms'] = target_component_synonyms

    i=0
    xref_src_db_list = []
    while(i <= len(all_xrefs)-1):
        if(all_xrefs[i].xref_src_db.string not in xref_src_db_list):
            xref_src_db_list.append(all_xrefs[i].xref_src_db.string)
        i += 1
    print("src list = ",xref_src_db_list)
    for j in xref_src_db_list:
        data_source = str(j)
        j = {}# 不同的数据库来源类型作不同的字典
        num=0
        xref_id = []
        xref_name = []
        while(num <= len(all_xrefs)-1):#循环结束会得到j（src name）对应的两个列表xref id表和xref name表
            if(all_xrefs[num].xref_src_db.string == data_source):
                xref_id.append(all_xrefs[num].xref_id.string)
                xref_name.append(all_xrefs[num].xref_name.string)
            num += 1
        j['xref_id'] = xref_id
        j['xref_name'] = xref_name
        temp_dict = copy.deepcopy(j)
        target_dict[data_source] = temp_dict



    target_dict['target_type'] = soup.target.target_type.string
    target_dict['tax_id'] = soup.target.tax_id.string


    temp_dict = copy.deepcopy(target_dict)
    target_list.append(temp_dict)
    count += 1

for element in target_list:#避免重复存取，只存储没有的，或更新现有的
    temp = dev2.update_one({'target_chembl_id': element['target_chembl_id']}, {'$set': element}, True)