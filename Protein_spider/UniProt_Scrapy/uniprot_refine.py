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
# 建立mongodb连接
mongoClient = MongoClient('mongodb://localhost:27017/')
#mongoClient = MongoClient('mongodb://172.10.10.9:27019/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
#dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
dev = dtmr_dev.get_collection('uniprot_TargetInfo_test')
dev2 = dtmr_dev.get_collection('uniprot_skip_error_id_list')

debug_UPIDs = ['A6H8Y1']  #'P05067','A0jNW5','A0jP26','A1X283','A2A2Y4'
skip_id_list = []
skip_id_dict = {}
records = []  # 创建空列表
no_topology = {}  # 创建空字典，其中topology为：蛋白结构域

#def returnlist():
'''
def returnlist(): 详情见generate_list.py:
该代码旨在通过pandas 读取uniprot_id.csv文件 后将其切片生成两个列表（uniprot_id_list and uniprot_name_list）
'''
data = pd.read_csv('uniprot_id.csv',engine='python')#sep='/n',
def return_id_name_list(data):
    i = 0#行指针
    j = 0#列指针(这个csv读取后生成的dataframe文件只有两列，一列id一列name)
    uniprot_id_list = []
    uniprot_name_list = []
    hang_len = data.shape[0]
    lie_len = data.shape[1]
    while(i <= hang_len-1 and j <= lie_len-1):
        uniprot_id_list.append(data.loc[i][j])
        i += 1#换行
        j += 1#换到第二列得到name
        uniprot_name_list.append(data.loc)
        j = 0 #列换回到第一列id列
    return uniprot_id_list,uniprot_name_list

uniprot_id_list,uniprot_name_list = return_id_name_list(data)

countid = 0
No = 0
pbar = ProgressBar()  # ProgressBar 用来显示进度条
for UPID in tqdm.tqdm(debug_UPIDs):#uniprot_id_list 的长度为20422，说明有两万多条uniprot id。 for UPID in tqdm.tqdm(uniprot_id_list):
    html_url = "https://rest.uniprot.org/uniprotkb/" + UPID  # uniprot数据存放不同于pdb，Uniprot要先获得数据包，用不了beautifulsoup对html源码进行解析。
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
    eval_error = 0
    try:
        eval(temp_dict)
    except Exception as error:
        print('错误类型是', error.__class__.__name__)
        print('错误明细是', error)
        eval_error = 1
    if(eval_error == 0):
        real_dict = eval(temp_dict)
        '''字符串转字典出现SyntaxError: EOL while scanning string literal   说明字符串中的引号和括号有未正确匹配成队'''
        #print('real_dict的数据类型为：', type(real_dict))
        '''此时不需要用beautifulsoup解析了直接对real_dict这个字典操作就行'''
        # soup = BeautifulSoup(html, 'html.parser')#解析器为html.parser
        countid += 1
        print('UPID:', UPID,"Count id = ",countid)
        record = {}  # 空字典record
        No += 1
        record["No"] = No  # 自增id NO
        record["UPID"] = UPID

        # overview = real_dict["content-protein"]#直接通过字典KEY值查找。
        # if overview == None:
        #     print('id:entry-overview抓取失败,没有这个keyword字段')
        #     break
        # record["Protein"] = overview.find("div", id="content-protein").text   #find().text指截取text内容
        # record["Gene"] = overview.find("div", id="content-gene").text
        # record["Organism"] = overview.find("div", id="content-organism").text  #organism：生物，物种
        record["TargetID"] = real_dict["primaryAccession"]
        record["Protein_english_name"] = real_dict["proteinDescription"]["recommendedName"]["fullName"]["value"]



        # Alternate_Name_list = real_dict["proteinDescription"]["alternativeNames"]  # 在json文件中一般来说每一个靶点蛋白对应的英文别名有多个
        if(bool(jsonpath(real_dict, '$..alternativeNames'))):
            list1 = jsonpath(real_dict, '$..alternativeNames')#$..proteinDescription.alternativeNames
            #print("bool?",bool(jsonpath(real_dict, '$..alternativeNames')))
            list2 = (list(_flatten(list1)))
            #print('Alternate_Name_list :',list2)
            num = 0
            Alternate_Name_list = []
            while(num <= len(list2)-1):
                Alternate_Name_list.append(list2[num]['fullName']['value'])
                num += 1
            record["Alternate_Name"] = Alternate_Name_list
        else:
            record["Alternate_Name"] = []

        # 可以像上面一样直接通过字典列表嵌套索引
        # 或者可以直接用jsonpath直接进行字典深度查询，不用自己逐层嵌套如上可以简化为：jsonpath(dict1, '$..value')[0] 即jsonpath后得到列表的第一个元素
        list1 = jsonpath(real_dict, '$.genes[*][*].value')  # 这句代码超级重要！！！！！连着两个[*][*]说明匹配下级元素的下级元素，只有这样才能找到我们想要的value
        # jsonpath 语法手册：https://www.jianshu.com/p/9808ab64fc0c
        # dict2=list2[0][0]
        # print("dict2=",dict2)
        # print("list1=",list1[0])
        record["Gene"] = list1[0]

        list1 = jsonpath(real_dict, '$..properties')  # 这个list1为列表类型
        list2 = list(_flatten(list1))  # 把二维列表转为一维，方便索引，当然也可以直接二维索引，不过那样更麻烦
        i = 0
        flag = 0
        PDB_name = real_dict["primaryAccession"]

        while (list2[i]["key"] != 'OrganismName'):
            i += 1
            if (i >= len(list2) - 1):
                record["Organism"] = '没有这个物种'
                flag = 1
                break
        # print(list2[i]["value"])
        # print("properties=",list2)
        if (flag == 0):
            record["Organism"] = list2[i]["value"]
        # count=0
        # while(list[count]["key"] != "OrganismName"):
        #     count += 1
        # if(jsonpath(real_dict, '$..properties')[0]["key"] =="OrganismName"):
        #     record["Organism"] = jsonpath(real_dict, '$..properties')[0]["value"]
        list1 = jsonpath(real_dict, '$..comments')
        list2 = (list(_flatten(list1)))
        i = 0
        flag = 0
        while (list2[i]["commentType"] != 'FUNCTION'):
            i += 1
            if (i >= len(list2) - 1):
                record["Fuction"] = "没有这种方法"
                flag = 1
                break
        # print("zhaodaolema?",list2[i]["texts"][0]["value"])
        if (flag == 0):
            record["Function"] = list2[i]["texts"][0]["value"]

        list1 = jsonpath(real_dict, '$..sequence')
        jason_dict = {PDB_name: list1[0]["value"]}
        record["Sequence"] = list1[0]["value"]
        # record["dict_Sequence"] = jason_dict


        list1 = jsonpath(real_dict, '$..evidences')
        list2 = (list(_flatten(list1)))
        # print("evidences:",list2)
        i = 0



        binding_site = []
        list1 = jsonpath(real_dict, '$..features')
        list2 = (list(_flatten(list1)))
        num = 0
        while(num <= len(list2)-1):
            if( str(list2[num]['type']) == "Binding site"):
                binding_site.append(list2[num])
            num += 1
        record["Binding_site"] = binding_site

        # Remark = {"remark": 'null'}
        # Relate_disease = {"relate": 'null'}
        # Binding_pocket = {"value": "null"}
        # Positive_compound = {"value": "null"}

        list1 = jsonpath(real_dict, '$..uniProtKBCrossReferences')
        list2 = (list(_flatten(list1)))
        # print("evidences:",list2)

        Structure = []

        uniprot_structure = {}

        num = 0
        while (num <= len(list2) - 1):
            if (bool(list2[num]['properties'])):  # 先看有没有properties这个字段
                if (bool(list2[num]['properties'][0]) and bool(
                        list2[num]['properties'][0]['key'])):  # 再看properties tag下的列表是不是空表，不是的话第一个元素有没有key
                    if (list2[num]['properties'][0]['key'] == "Method" and bool(
                            list2[num]['properties'][0]['value'])):  # 再看这个key下面的值是不是为"Method",且value不为空
                        # print('test:',list2[num]['database'])
                        # print('test:',list2[num]['id'])
                        # print('type:',uniprot_structure)
                        uniprot_structure['SOURCE'] = list2[num]['database']
                        uniprot_structure['IDENTIFIER'] = list2[num]['id']
                        uniprot_structure['METHOD'] = list2[num]['properties'][0]['value']
                if (len(list2[num]['properties']) >= 2):  # 有可能properties下面就只有上面一条，判断一下列表有没有第二个元素
                    if (list2[num]['properties'][1]['key'] == "Resolution" and bool(
                            list2[num]['properties'][1]['value'])):  # 再看这个key下面的值是不是为"Resolution",且value不为空
                        uniprot_structure['Resolution'] = list2[num]['properties'][1]['value']
                if (len(list2[num]['properties']) >= 3):
                    if (list2[num]['properties'][2]['key'] == "Chains"):
                        uniprot_structure['CHAIN&POSITION'] = list2[num]['properties'][2]['value']
            temp_uniprot_structure = copy.deepcopy(uniprot_structure)
            if (temp_uniprot_structure):  # temp_uniprot_structure为空字典就不用插了 在python里，{},[],()，等都等价于False！
                Structure.append(temp_uniprot_structure)
            #print('test=', temp_uniprot_structure)
            uniprot_structure = {}
            num += 1

        record["Structure"] = Structure

        record["Relate_disease"] = 'null'
        record["Binding_pocket"] = 'null'
        record["Postivecompound"] = 'null'
        record["Remark"] = 'null'
        record["Topology"] = 'null'
    else:#没办法通过eval() 将json转字典了,那就只能读json文件了,这里先跳过，以后再加
        #real_dict = json.loads(temp_dict)
        No += 1
        record = {}
        print("skip forward")
        skip_id_list.append(UPID)
        record["No"] = No  # 自增id NO
        record["UPID"] = UPID
        skip_id_dict["error_upid"] = skip_id_list
        temp_dict = copy.deepcopy(skip_id_dict)
        dev2.insert_one(temp_dict)


    records.append(record)


result = dev.insert_many(records)#mongodb insert_many
print(len(result.inserted_ids), 'records inserted!')




