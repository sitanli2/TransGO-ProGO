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
mongoClient = MongoClient('mongodb://localhost:27017/')
dtmr_dev = mongoClient.get_database('Chembl_DOC')
dev2 = dtmr_dev.get_collection('Target')

'''activities 活性物接口爬取'''
activities_id_list = ['CHEMBL403233']
activities_list = []
count = 0

for i in activities_id_list:
    print('此时的activities id=', i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')

    activitise_dict = {}
    ligand = {}  # 作为活性物字典activitise_dict 中的ligand嵌套字典
    activitise_dict['activity_id'] = soup.molecule.molecule_chembl_id.string
    activitise_dict['Compound Key'] = soup.molecule.molecule_properties.rtb.string
    activitise_dict['Standard Type'] = soup.molecule.molecule_properties.rtb.string


    # html_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit=5"  # 直接拿它给的前5条activity
    # response = request.urlopen(html_url)
    # XML = response.read()
    # soup = BeautifulSoup(XML, 'xml')
    # all_activities = soup.activities.find_all('activity', recursive=False)
    # # all_activities 是一个长度为五的列表，列表元素类型为 <class 'bs4.element.Tag'> 所以可以对列表中的每个元素进行bs解析。

    activitise_dict['activity_id'] = all_activities[i].activity_id.string
    activitise_dict['target_chembl_id'] = all_activities[i].target_chembl_id.string
    activitise_dict['target_tax_id'] = all_activities[i].target_tax_id.string
    activitise_dict['assay_chembl_id'] = all_activities[i].assay_chembl_id.string

    activitise_dict['document_chembl_id'] = all_activities[i].document_chembl_id.string
    activitise_dict['molecule_chembl_id'] = all_activities[i].molecule_chembl_id.string
    activitise_dict['parent_molecule_chembl_id'] = all_activities[i].parent_molecule_chembl_id.string
    activitise_dict['record_id'] = all_activities[i].record_id.string

    activitise_dict['target_organism'] = all_activities[i].target_organism.string
    activitise_dict['target_pref_name'] = all_activities[i].target_pref_name.string
    activitise_dict['assay_description'] = all_activities[i].assay_description.string
    activitise_dict['assay_type'] = all_activities[i].assay_type.string
    activitise_dict['bao_endpoint'] = all_activities[i].bao_endpoint.string
    activitise_dict['bao_format'] = all_activities[i].bao_format.string
    activitise_dict['bao_label'] = all_activities[i].bao_label.string
    activitise_dict['type'] = all_activities[i].type.string
    if (all_activities[i].ligand_efficiency.bei != None or all_activities[i].ligand_efficiency.le != None):
        print('ligand output:', all_activities[i].ligand_efficiency.bei)
        ligand['bei'] = all_activities[i].ligand_efficiency.bei.string
        ligand['le'] = all_activities[i].ligand_efficiency.le.string
        ligand['lle'] = all_activities[i].ligand_efficiency.lle.string
        ligand['sei'] = all_activities[i].ligand_efficiency.sei.string

    activitise_dict['ligand_effieiency'] = ligand
    temp_dict = copy.deepcopy(activitise_dict)
    activities_list.append(temp_dict)
    count += 1
dev2.insert_many(activities_list)














