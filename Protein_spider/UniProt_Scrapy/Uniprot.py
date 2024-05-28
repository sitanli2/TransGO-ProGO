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
import  ast
from jsonpath import jsonpath
from tkinter import _flatten #将二维列表转成一维列表
# 建立mongodb连接
mongoClient = MongoClient('mongodb://localhost:27017/')
# mongoClient = MongoClient('mongodb://172.10.10.9:27019/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
# dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
dev = dtmr_dev.get_collection('uniprot_TargetInfo')

# 创建GridFS对象，传入参数一：数据库，参数二：集合
# gfs_af = GridFS(dtmr_dev, collection='AlphafoldModel_files')
# gfs_fasta = GridFS(dtmr_dev, collection='FASTA_files')

'''
#UPIDs = pd.read_excel("target_ID.xlsx").sort_values(by="No.").set_index("No.").Human_UPID.values
#panda.read_excel读取表格数据
pandas中的sort_values()函数原理类似于SQL中的order by，可以将数据集依照某个字段中的数据进行排序，该函数即可根据指定列数据也可根据指定行的数据排序。
这个例子我直接给出UPID的列表
'''
UPIDs =['P05067']#'A0jNW5','A0jP26','A1X283','A2A2Y4'
records = []#创建空列表
no_topology = {}#创建空字典，其中topology为：蛋白结构域

No = 0
pbar = ProgressBar()#ProgressBar 用来显示进度条
for UPID in  tqdm.tqdm(UPIDs):
    html_url = "https://rest.uniprot.org/uniprotkb/"+UPID #uniprot数据存放不同于pdb，Uniprot要先获得数据包，用不了beautifulsoup对html源码进行解析。
    # https: // rest.uniprot.org / uniprotkb / P05067  为什么访问这个网址看最下面的注释。
    #html_url = "https://www.uniprot.org/uniprotkb/" + UPID  这个网页的HTML无法爬取，所以之前学姐留下的代码是没有用的
    response = request.urlopen(html_url)
    html = response.read()
    print("html:",html)
    temp_dict = str(html).split("b'")[1].strip("'")
    print("html after split:",temp_dict)
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
    #real_dict = json.loads(temp_dict)
    #real_dict = ast.literal_eval(temp_dict)
    global false,null,true
    false=null=true=''#由于eval() 不能处理false，因此要先将false设为全局变量，并将其置空
    real_dict=eval(temp_dict)
    '''字符串转字典出现SyntaxError: EOL while scanning string literal   说明字符串中的引号和括号有未正确匹配成队'''
    print('real_dict的数据类型为：', type(real_dict))
    '''此时不需要用beautifulsoup解析了直接对real_dict这个字典操作就行'''
    #soup = BeautifulSoup(html, 'html.parser')#解析器为html.parser
    print('UPID:',UPID)
    record = {}#空字典record
    No += 1
    record["No"] = No#自增id NO
    record["UPID"] = UPID

    # overview = real_dict["content-protein"]#直接通过字典KEY值查找。
    # if overview == None:
    #     print('id:entry-overview抓取失败,没有这个keyword字段')
    #     break
    # record["Protein"] = overview.find("div", id="content-protein").text   #find().text指截取text内容
    # record["Gene"] = overview.find("div", id="content-gene").text
    # record["Organism"] = overview.find("div", id="content-organism").text  #organism：生物，物种
    record["targetID"] = real_dict["primaryAccession"]
    record["Protein_english_name"]=real_dict["proteinDescription"]["recommendedName"]["fullName"]["value"]
    #可以像上面一样直接通过字典列表嵌套索引
    #或者可以直接用jsonpath直接进行字典深度查询，不用自己逐层嵌套如上可以简化为：jsonpath(dict1, '$..value')[0] 即jsonpath后得到列表的第一个元素
    list1=jsonpath(real_dict, '$.genes[*][*].value')#这句代码超级重要！！！！！连着两个[*][*]说明匹配下级元素的下级元素，只有这样才能找到我们想要的value
    #jsonpath 语法手册：https://www.jianshu.com/p/9808ab64fc0c
    # dict2=list2[0][0]
    # print("dict2=",dict2)
    #print("list1=",list1[0])
    record["Gene"] = list1[0]
    list1=jsonpath(real_dict, '$..properties')#这个list1为列表类型
    list2=list(_flatten(list1))#把二维列表转为一维，方便索引，当然也可以直接二维索引，不过那样更麻烦
    i = 0
    flag = 0
    PDB_name = real_dict["primaryAccession"]


    while(list2[i]["key"] != 'OrganismName'):
        i += 1
        if(i >= len(list2)-1 ):
            record["Organism"]= '没有这个物种'
            flag=1
            break
    #print(list2[i]["value"])
    #print("properties=",list2)
    if (flag == 0):
        record["Organism"] = list2[i]["value"]
    # count=0
    # while(list[count]["key"] != "OrganismName"):
    #     count += 1
    # if(jsonpath(real_dict, '$..properties')[0]["key"] =="OrganismName"):
    #     record["Organism"] = jsonpath(real_dict, '$..properties')[0]["value"]
    list1 = jsonpath(real_dict, '$..comments')
    list2=(list(_flatten(list1)))
    i = 0
    flag=0
    while (list2[i]["commentType"] != 'FUNCTION'):
        i += 1
        if (i >= len(list2)-1 ):
            record["Fuction"] = "没有这种方法"
            flag=1
            break
    #print("zhaodaolema?",list2[i]["texts"][0]["value"])
    if(flag == 0):
        record["Function"] = list2[i]["texts"][0]["value"]

    list1 = jsonpath(real_dict, '$..sequence')
    jason_dict={PDB_name : list1[0]["value"]}
    record["Sequence"] = list1[0]["value"]
    #record["dict_Sequence"] = jason_dict
    record["Topology"] = null

    list1 = jsonpath(real_dict, '$..evidences')
    list2 = (list(_flatten(list1)))
    #print("evidences:",list2)
    i = 0
    Structure = {}
    EvidenceCode = {}
    Source = {}
    Identifier = {}



    while (type(list2[i]["evidenceCode"]) == str ):
        EvidenceCode["code"] = list2[i].get('evidenceCode')
        #print("source的类型：",type(list2[i].get('source')))
        '''之前用list2[i]['source']方法查询字典'source'和id出现了'keyerror'，换用dict.get()就没有了'''


        # if (type(list2[i].get('source')) == str and len(list2[i].get('id')) <= 6):
        #      Source["source"] = list2[i].get('source')
        # else:
        #      Source["source"] = "source为空"
        # if (type(list2[i].get('id')) == str and len(list2[i].get('id')) <= 6):
        #      Identifier["ID"] = list2[i].get('id')
        # else:
        #      Identifier["ID"] = "ID为空"
        i += 1
        if (i >= len(list2) - 1):
            break
    i=0
    while (type(list2[i]["evidenceCode"]) == str):
        if (type(list2[i].get('source')) == str and len(list2[i].get('id')) <= 5 ):
            Source["source"] = list2[i].get('source')
            Identifier["ID"] = list2[i].get('id')
            break
        i += 1
        if (i >= len(list2) - 1):
            Source["source"] = "source 为空"
            break

    Structure_add = {}
    Method = {}
    Resolution = {}
    Chain = {}
    Position = {}
    Remark = {"remark":'null'}
    Relate_disease = {"relate":'null'}
    Binding_pocket = {"value":"null"}
    Positive_compound = {"value":"null"}
    record["Relate_disease"] = Relate_disease
    record["binding"] = Binding_pocket
    record["Postivecompound"] = Positive_compound
    record["Remark"] = Remark


    list1 = jsonpath(real_dict, '$..properties')
    list2 = (list(_flatten(list1)))
    #print("properties:",list2)
    i=0
    while (list2[i].get("key") != 'Method'):
        i += 1
        if (i >= len(list2) - 1):
            break
    if (i == len(list2) - 1):
        Method["method"] = '方法为空'
        print('此时i=',i)
    else:
        Method["method"] = list2[i].get('value')
    i=0
    while (list2[i].get("key") != 'Resolution'):
        i += 1
        if (i >= len(list2) - 1):
            break
    if (i == len(list2) - 1):
        Resolution["resolve"] = 'resolution 为空'
    else:
        Resolution["resolve"] = list2[i].get('value')

    i=0
    while (list2[i].get("key") != 'Chains'):
        i += 1
        if (i >= len(list2) - 1):
            break
    if (i == len(list2) - 1):
        Chain["chain"] = 'chain 为空'
    else:
        Chain["chain"] = list2[i].get('value')



    Structure_add["METHOD"] = Method
    Structure_add["RESOLUTION"] = Resolution
    Structure_add["CHAIN&POSITION"] = Chain
    Structure["SOURCE"] = Source
    Structure["IDENTIFIER"] = Identifier
    Structure["EVIDENCECODE"] = EvidenceCode
    Structure["add_item"] = Structure_add
    record["Structure_uniprot"] = Structure




    # function_list = html.find("div", id="function")#找到tag为<div...的第一个id="fuction"的字段
    # function = {}#空字典function
    # if function_list == None:
    #     print('id="function"抓取失败，没有这个字段')
    #     break
    # for ul in function_list.find_all("ul", class_='noNumbering'):
    #     if ul.attrs != {}:
    #         name = ul.attrs["class"][1]
    #         function[name] = []
    #         for li in ul.find_all("li"):
    #             if li.attrs != {}:
    #                 pair = {}
    #                 pair["GO"] = li.attrs["class"][0]
    #                 pair["text"] = li.find("a").text
    #                     #                     print(link, text)
    #                 function[name].append(pair)
    #
    # record["Function"] = function#嵌套字典，即在record中在插入一个keyword:Fuction，内容为上述fuction{}的一个字典
    # no_topology = {}#创建空字典
    # topology_table = html.find("table", id="topology_section")
    # if topology_table == None:
    #     record["Topology"] = None
    #     no_topology[UPID] = record["Protein"]
    # else:
    #     topology = []
    #     for tr in topology_table.find_all("tr"):
    #         row = {}
    #         for td in tr.find_all("td")[:3]:
    #             if td != []:
    #                 fkey = td.find("span", class_="context-help")
    #                 if fkey != None:
    #                     for s in fkey.select('span'):
    #                         s.extract()
    #                     if fkey != None:
    #                         row["Feature key"] = fkey.text
    #
    #                 posi = td.find("a", class_="position tooltipped")
    #                 if posi != None:
    #                     posi.text
    #                     row["Positions"] = tuple(int(i) for i in posi.text.split("\xa0–\xa0"))
    #
    #                 desp = td.find("span", property="text")
    #                 if desp != None:
    #                     row["Description"] = desp.text
    #         if row != {}:
    #             topology.append(row)
    #
    #     record["Topology"] = topology#record列表创建新列表项topology：蛋白结构域

    # r = requests.get('https://alphafold.ebi.ac.uk/files/AF-' + UPID + '-F1-model_v1.pdb', allow_redirects=True)
    # if r.status_code != 200:
    #     record["AlphafoldModel"] = None
    # else:
    #     record["AlphafoldModel"] = gfs_af.put(r.content, filename=UPID+".pdb", type="pdb", keyword=UPID)
    #
    # r = requests.get("https://www.uniprot.org/uniprot/" + UPID + ".fasta", allow_redirects=True)
    # if r.status_code != 200:
    #     record["FASTA"] = None
    # else:
    #     record["FASTA"] = gfs_fasta.put(r.content, filename=UPID+".fasta", type="fasta", keyword=UPID)

    records.append(record)

result = dev.insert_many(records)#mongodb insert_many
print(len(result.inserted_ids), 'records inserted!')

# with open('UniProt_data.json', 'w', encoding="utf-8") as fp:
#     json.dump(records, fp, ensure_ascii=False)


# with open('UniProt_no_topology.json', 'w', encoding="utf-8") as fp:
#     json.dump(no_topology, fp, ensure_ascii=False)

'''
1.Uniprot网页 爬虫 继续debug

由于某些原因，不同于PDB网站，uniprot网页的HTML源码无法用beautifulsoup的html.parser解析（有可能我的浏览器版本不行或者Uniprot网页有反爬虫插件）所以只能另寻他法：
首先访问uniprot 网站，搜索蛋白质id：P0567 -> 微软Edge浏览器右上角三点 -> 更多工具 -> 开发人员工具 -> 网络 -> ctrl+r刷新网络记录 -> 全部 ->
找到我们要找的数据包P05067 -> 再在“标头”选项找到：请求方法get对应的请求网址URL：https://rest.uniprot.org/uniprotkb/

接下来我们就可以通过这个新网址得到我们要找的P05067 蛋白质在uniprot网站下的数据包（是一个很大的字典，但是数据类型是字符串，需要我们之后将其转成字典）
html_url = "https://rest.uniprot.org/uniprotkb/"+ UPID #这里的UPID就是P05067
response = request.urlopen(html_url)
html = response.read()
此时我们的html是一个字节型文件（byte class），再将其转换为字符串，方便后续的转字典操作：

str.split() 方法被用于通过指定分隔符对字符串进行切片：str.split(str="", num=string.count(str)) 若参数num（分割次数）有指定值，则分割num+1个字符串。
最后的返回值是一个被分割后的字符串列表
由上可知 temp_dict = str(html).split("b'")[1] 在于去除字符串html前面的"b'"
此时temp_dict是一个字符串类型的文件还要将其转成字典，详情见同目录下Uniprot.py代码文件中的注释。

strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

2.字符串转换成字典：获得的蛋白质数据包是一个列表嵌套字典的string型文件，因此为方便后续便利必须: String -> 二维List -> 一维List -> dict
2.1 字符串转字典时出现错误1 SyntaxError: EOL while scanning string literal   说明字符串中的引号和括号有未正确匹配成队
要通过修改字符串将其改为合法字典

2.2 在将字符串转换为字典时，eval不能解析false所以会报错
因该将false设为全局变量然后将其置空：
global false,null,true
false=null=true=''
2.3在将字符串转换为字典时 ,用real_dict = json.loads(temp_dict)方法进行转换需注意temp_dict字符串中全部为双引号，否则也会报错。

3.用记事本存放dict Ctrl+f查找对应关键字，方便继续索引
也可以直接通过 浏览器在uniprot网站 搜索要爬取的蛋白质 id：P0567 -> 微软Edge浏览器右上角三点 -> 更多工具 -> 开发人员工具 -> 网络 -> ctrl+r刷新网络记录 -> 全部 ->
找到我们要找的数据包P05067 -> 预览  则可以看到完整的字典树，从而再回去从根节点开始索引要找到的“叶子”

4.使用jsonpath,深度查找定位字典对应的keyword（#jsonpath 语法手册：https://www.jianshu.com/p/9808ab64fc0c）
如list1=jsonpath(real_dict, '$.genes[*][*].value')#这句代码超级重要！！！！！连着两个[*][*]说明匹配下级元素的下级元素，只有这样才能快速找到我们想要的value
由于jsonpath得到的是一个列表，所以之后为了方便继续索引列表：
list2=list(_flatten(list1))#把二维列表转为一维，方便索引，当然也可以直接二维索引，不过那样更麻烦
最后总算得到我们要找的字典，再通过{keyword : value}来得到最终结果插入MongoDB

5.之前用list2[i]['source']方法查询字典'source'和'id'出现了'keyerror'，但是显然字典内是存在source和id这两个keyword的，
可是之后换用dict.get()就没有报错了了如：list2[i].get('source')便不会有keyword error报错

*以上项目详情见Uniprot.py 代码
'''