from pymongo import MongoClient
from gridfs import GridFS
import pandas as pd
import tqdm
import requests
import urllib.request as request #urllib3版本要<=1.25否则会与requests版本冲突
from bs4 import BeautifulSoup
from progressbar import ProgressBar
import json
import  ast
from jsonpath import jsonpath
from tkinter import _flatten #将二维列表转成一维列表
import copy

mongoClient = MongoClient('mongodb://localhost:27017/')
# mongoClient = MongoClient('mongodb://172.10.10.9:27019/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
# dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
dev = dtmr_dev.get_collection('uniprot_TargetInfo')

UPIDs = ['P05067']  # 'A0jNW5','A0jP26','A1X283','A2A2Y4'
records = []  # 创建空列表
record = {}

html_url = "https://rest.uniprot.org/uniprotkb/" + 'P05067'  # uniprot数据存放不同于pdb，Uniprot要先获得数据包，用不了beautifulsoup对html源码进行解析。
# https: // rest.uniprot.org / uniprotkb / P05067  为什么访问这个网址看最下面的注释。
# html_url = "https://www.uniprot.org/uniprotkb/" + UPID  这个网页的HTML无法爬取，所以之前学姐留下的代码是没有用的
response = request.urlopen(html_url)
html = response.read()
#print("html:", html)
temp_dict = str(html).split("b'")[1].strip("'")
#print("html after split:", temp_dict)
'''
str.split() 方法被用于通过指定分隔符对字符串进行切片：str.split(str="", num=string.count(str)) 若参数num（分割次数）有指定值，则分割num+1个字符串。
最后的返回值是一个被分割后的字符串列表
由上可知 temp_dict = str(html).split("b'")[1] 在于去除字符串html前面的"b'" 

strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
'''
# json库有以下两个功能：
# json.dumps()：
# 用来接收一个Python对象，并将其转换为字符串；
# json.loads()：
# 接收JSON字符串，并将其转换为Python对象。
# real_dict = json.loads(temp_dict)
# real_dict = ast.literal_eval(temp_dict)
global false, null, true
false = null = true = ''  # 由于eval() 不能处理false，因此要先将false设为全局变量，并将其置空
real_dict = eval(temp_dict)

list1 = jsonpath(real_dict, '$..uniProtKBCrossReferences')
list2 = (list(_flatten(list1)))
# print("evidences:",list2)

Structure = []

uniprot_structure = {}

num = 0
while(num <= len(list2)-1):
    if(bool(list2[num]['properties']) ):#先看有没有properties这个字段
        if(bool(list2[num]['properties'][0]) and bool(list2[num]['properties'][0]['key'])):#再看properties tag下的列表是不是空表，不是的话第一个元素有没有key
            if(list2[num]['properties'][0]['key'] == "Method" and bool(list2[num]['properties'][0]['value'])):#再看这个key下面的值是不是为"Method",且value不为空
                # print('test:',list2[num]['database'])
                # print('test:',list2[num]['id'])
                # print('type:',uniprot_structure)
                uniprot_structure['SOURCE'] = list2[num]['database']
                uniprot_structure['IDENTIFIER'] = list2[num]['id']
                uniprot_structure['METHOD'] = list2[num]['properties'][0]['value']
        if(len(list2[num]['properties']) >= 2):#有可能properties下面就只有上面一条，判断一下列表有没有第二个元素
            if(list2[num]['properties'][1]['key'] == "Resolution" and bool(list2[num]['properties'][1]['value'])):#再看这个key下面的值是不是为"Resolution",且value不为空
                uniprot_structure['Resolution'] = list2[num]['properties'][1]['value']
        if(len(list2[num]['properties']) >= 3):
            if(list2[num]['properties'][2]['key'] == "Chains"):
                uniprot_structure['CHAIN&POSITION'] = list2[num]['properties'][2]['value']
    temp_uniprot_structure = copy.deepcopy(uniprot_structure)
    if(temp_uniprot_structure):#temp_uniprot_structure为空字典就不用插了 在python里，{},[],()，等都等价于False！
        Structure.append(temp_uniprot_structure)
    print('test=',temp_uniprot_structure)
    uniprot_structure = {}
    num += 1

record["Structure"] = Structure
dev.insert_one(record)




