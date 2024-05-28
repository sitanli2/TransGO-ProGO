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
import copy
from tkinter import _flatten #将二维列表转成一维列表
import time
import os


# 建立mongodb连接
mongoClient = MongoClient('mongodb://localhost:27017/')
# mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
# dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev') #密码验证

dev = dtmr_dev.get_collection('uniprot_Target_Info') #uniprot_Target_Info_test
dev2 = dtmr_dev.get_collection('uniprot_skip_error_id_list')
devlog = dtmr_dev.get_collection('uniprot_Log')

debug_UPIDs = ['P05067']  #'P05067','A0jNW5','A0jP26','A1X283','A2A2Y4'
skip_id_list = []
skip_id_dict = {}
records = []  # 创建空列表
no_topology = {}  # 创建空字典，其中topology为：蛋白结构域

#def returnlist():
'''
def returnlist(): 详情见generate_list.py:
该代码旨在通过pandas 读取uniprot_id.csv文件 后将其切片生成两个列表（uniprot_id_list and uniprot_name_list）
'''

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

def MongoConnection():
    mongoClient = MongoClient('mongodb://localhost:27017/')
    database = mongoClient.get_database('dtmr_dev')

    # collection = dtmr_dev['uniprot_Target_Info_test']

    collection = database.get_collection('uniprot_Target_Info_test')
    return collection

def getEntryIDFromMongo_ThenDownload(UPID,save_path):
    collection = MongoConnection()
    try:
        query_result = collection.find_one({"UPID": UPID})["Structure_AlphaFold"]["IDENTIFIER_entryId"]  # 从MonGoDB中通过UPID索引到其对应条目的 Structure_AlphaFold 嵌套字典
        print(UPID+"的IDENTIFIER_entryId 查询结果:", query_result,'\n返回类型：', type(query_result))
        entryID = query_result
    except:
        print("查询不到UPID=",UPID,"的索引条目，没有与之对应的AlphaFold预测结构")



    # 先要使用函数getEntryIDFromMongo根据UPID（如 A0A087X1C5）从MongoDB中获取到entryID（AF-A0A087X1C5-F1）
    # 通过这个entryID(AF-A0A087X1C5-F1)去到AlpaFold数据接口：https://alphafold.ebi.ac.uk/files/AF-A0A087X1C5-F1-model_v4.pdb
    # 把对应的PDB文件Down下来并放到目标路径中去（并以UPID命名 即 A0A087X1C5.pdb）
    os.makedirs(save_path, exist_ok=True)  # 确保目标路径存在
    DownloadURL = "https://alphafold.ebi.ac.uk/files/" + entryID + "-model_v4.pdb"
    #再看我之前的23年实习工作报告时，注意到另外一个接口:
    # "https://alphafold.ebi.ac.uk/api/prediction/" + UPID  # 这个alphafold prediction 接口请求方法也为get 用来通过UPID来获取alphafold，但是这个接口不提供对应PDB文件的下载。
    filename = UPID+".pdb"
    response = requests.get(DownloadURL)
    if response.status_code == 200:
        # 构建保存路径
        save_path = os.path.join(save_path, filename)

        # 写入文件
        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"文件已保存至：{save_path}")
    else:
        print("下载失败")




# def DownloadPDBFromAlphaFold(entryID,save_path):






def mainscrapy():
    data = pd.read_csv('uniprot_id.csv', engine='python')  # sep='/n',
    uniprot_id_list,uniprot_name_list = return_id_name_list(data)
    uniprot_id_list = ['P05067'] #这个是用来测试的
    No = 0
    pbar = ProgressBar()  # ProgressBar 用来显示进度条
    startpoint = input("输入你上次的断点位置=")#输入值就为log里面的断点值

    countid = int(startpoint)


    for UPID in tqdm.tqdm( uniprot_id_list[int(startpoint):] ):#uniprot_id_list 的长度为20422，说明有两万多条uniprot id。 for UPID in tqdm.tqdm(uniprot_id_list):
        html_url = "https://rest.uniprot.org/uniprotkb/" + UPID  # uniprot数据存放不同于pdb，Uniprot要先获得数据包，用不了beautifulsoup对html源码进行解析。
        # https: // rest.uniprot.org / uniprotkb / P05067  为什么访问这个网址看最下面的注释。
        # html_url = "https://www.uniprot.org/uniprotkb/" + UPID  这个网页的HTML现在无法直接通过前端爬取，所以之前留下的代码是没有用的

        #html_url_prediction = "https://alphafold.ebi.ac.uk/api/prediction/P05067" #alphafold prediction 接口请求方法也为get 用来获取alphafold

        # response = request.urlopen(html_url)
        """
        直接用 response = request.urlopen(html_url)  请求接口的话
        有时候可能会发生断网从而导致程序中断，所以要加个判错，让他在网络连接成功之前都一直请求下去
        """
        while True:
            try:
                response = request.urlopen(html_url)  # 请求成功
                break
            except Exception as e:
                # 如果发生异常，则打印异常信息，并等待一段时间后重新尝试请求
                print("网页请求失败:", e)
                print("等待5秒后将重新尝试请求...")
                time.sleep(5)  # 等待5秒后重新尝试请求

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
        if(eval_error == 0): #未发生eval错误
            real_dict = eval(temp_dict)
            '''字符串转字典出现SyntaxError: EOL while scanning string literal   说明字符串中的引号和括号有未正确匹配成队'''
            #print('real_dict的数据类型为：', type(real_dict))
            '''此时不需要用beautifulsoup解析了直接对real_dict这个字典操作就行'''
            # soup = BeautifulSoup(html, 'html.parser')#解析器为html.parser
            countid += 1
            record = {}  # 空字典record

            list1 = jsonpath(real_dict, '$..inactiveReasonType')#有id是属于inactive状态，可能已经被删掉了，所以得给他另外存
            #print("inactive id ?", list1)
            if(list1):#list1 为True 说明他确实是inactive的id
                #record["No"] = No
                record["UPID"] = UPID
                record["inactive"] = 'True'
                record["inactiveReasonType"] = list1
                No += 1
                result = dev.insert_one(record)
                print( 'inactive records inserted!')
                continue
            print('UPID:', UPID, "Count id = ", countid)
            startpoint = countid
            log = {"最近的断点位置":startpoint}
            devlog.insert_one(log)
            No += 1
            #record["No"] = No  # 自增id NO
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
                record["Alternate_Name"] = 'null'

            # 可以像上面一样直接通过字典列表嵌套索引
            # 或者可以直接用jsonpath直接进行字典深度查询，不用自己逐层嵌套如上可以简化为：jsonpath(dict1, '$..value')[0] 即jsonpath后得到列表的第一个元素
            #list1 = jsonpath(real_dict, '$.genes[*][*].value')  # 这句代码超级重要！！！！！连着两个[*][*]说明匹配下级元素的下级元素，只有这样才能找到我们想要的value
            # jsonpath 语法手册：https://www.jianshu.com/p/9808ab64fc0c
            # dict2=list2[0][0]
            # print("dict2=",dict2)
            # print("list1=",list1[0])
            list1 = jsonpath(real_dict,'$..genes')


            if (bool(list1)):
                list2 = list(_flatten(list1))
                # if (bool(list2[0]['geneName'])): #当 UPID: P09565 时，出现错误  KeyError: 'geneName'
                #print("UPID= ",UPID,"时list2 = ",list2) #debug UPID: P09565
                try:
                    record["Gene"] = list2[0]['geneName']['value']
                except Exception as error:
                    print('错误类型是', error.__class__.__name__)
                    print('错误明细是', error)
                    record["Gene"] = list2[0]['orfNames'] #有两个 geneName 别名？

            else:
                record["Gene"] = 'null'

            list1 = jsonpath(real_dict,'$..genes[0].synonyms')
            #print("gene synonyms test=", list2)
            if(list1):
                list2 = list(_flatten(list1))
                count = 0
                synonyms_list = []
                while(count <= len(list2)-1):
                    # synonyms_dict = {}
                    # synonyms_dict = list2[count]['value']
                    # tempdict = copy.deepcopy(synonyms_dict)
                    synonyms_list.append(list2[count]['value'])
                    count += 1
            else:
                print("该蛋白质gene没有别名")
                synonyms_list = []
            record["Gene_synonyms"] = synonyms_list

            list1 = jsonpath(real_dict, '$..properties')  # 这个list1为列表类型
            if(list1):
                list2 = list(_flatten(list1))  # 把二维列表转为一维，方便索引，当然也可以直接二维索引，不过那样更麻烦
                i = 0
                flag = 0
                PDB_name = real_dict["primaryAccession"]

                while (list2[i]["key"] != 'OrganismName'):
                    i += 1
                    if (i >= len(list2) - 1):
                        record["Organism"] = 'null'
                        flag = 1
                        break
                # print(list2[i]["value"])
                # print("properties=",list2)
                if (flag == 0):
                    record["Organism"] = list2[i]["value"]
            else:
                record["Organism"] = 'null'
            # count=0
            # while(list[count]["key"] != "OrganismName"):
            #     count += 1
            # if(jsonpath(real_dict, '$..properties')[0]["key"] =="OrganismName"):
            #     record["Organism"] = jsonpath(real_dict, '$..properties')[0]["value"]
            list1 = jsonpath(real_dict, '$..comments')
            #print("debug Function:",list1)
            if(list1):
                list2 = (list(_flatten(list1)))
                i = 0
                flag = 0
                while (list2[i]["commentType"] != 'FUNCTION'):
                    i += 1
                    if (i >= len(list2) - 1):
                        record["Function"] = "null"
                        flag = 1
                        break
                # print("zhaodaolema?",list2[i]["texts"][0]["value"])
                if (flag == 0):
                    record["Function"] = list2[i]["texts"][0]["value"]
            else:
                record["Function"] = 'null'

            list1 = jsonpath(real_dict, '$..sequence')
            #print("debug sequence:",list1)
            if(list1):
                jason_dict = {PDB_name: list1[0]["value"]}
                record["Sequence"] = list1[0]["value"]
            # record["dict_Sequence"] = jason_dict
            else:
                record["Sequence"] = 'null'

            #
            # list1 = jsonpath(real_dict, '$..evidences')
            # list2 = (list(_flatten(list1)))
            # # print("evidences:",list2)
            # i = 0



            binding_site = []
            list1 = jsonpath(real_dict, '$..features')
            #print("debug bindingsite:",list1)
            if(list1):
                list2 = (list(_flatten(list1)))
                num = 0
                while(num <= len(list2)-1):
                    if( str(list2[num]['type']) == "Binding site"):
                        binding_site.append(list2[num])
                    num += 1
                record["Binding_site"] = binding_site
            else:
                record["Binding_site"] = 'null'

            # Remark = {"remark": 'null'}
            # Relate_disease = {"relate": 'null'}
            # Binding_pocket = {"value": "null"}
            # Positive_compound = {"value": "null"}


            list1 = jsonpath(real_dict, '$..uniProtKBCrossReferences')
            #print("debug structure:",list1)
            if(list1):
                list2 = (list(_flatten(list1)))
                #print("evidences:",list2)

                Structure = []

                # uniprot_structure = {}

                num = 0
                flag = 0#用来监测是否有alphafold
                while (num <= len(list2) - 1):
                    # if (bool(list2[num]['properties'])):  # 先看有没有properties这个字段
                    #     if (bool(list2[num]['properties'][0]) and bool(list2[num]['properties'][0]['key'])):  # 再看properties tag下的列表是不是空表，不是的话第一个元素有没有key
                    #         if (list2[num]['properties'][0]['key'] == "Method" and bool(list2[num]['properties'][0]['value'])):  # 再看这个key下面的值是不是为"Method",且value不为空
                    #             # print('test:',list2[num]['database'])
                    #             # print('test:',list2[num]['id'])
                    #             # print('type:',uniprot_structure)
                    #             uniprot_structure['SOURCE'] = list2[num]['database']
                    #             uniprot_structure['IDENTIFIER'] = list2[num]['id']
                    #             uniprot_structure['METHOD'] = list2[num]['properties'][0]['value']
                    #     if (len(list2[num]['properties']) >= 2):  # 有可能properties下面就只有上面一条，判断一下列表有没有第二个元素
                    #         if (list2[num]['properties'][1]['key'] == "Resolution" and bool(list2[num]['properties'][1]['value'])):  # 再看这个key下面的值是不是为"Resolution",且value不为空
                    #             uniprot_structure['Resolution'] = list2[num]['properties'][1]['value']
                    #     if (len(list2[num]['properties']) >= 3):
                    #         if (list2[num]['properties'][2]['key'] == "Chains"):
                    #             uniprot_structure['CHAIN&POSITION'] = list2[num]['properties'][2]['value']
                    #
                    # temp_uniprot_structure = copy.deepcopy(uniprot_structure)
                    # if (temp_uniprot_structure):  # temp_uniprot_structure为空字典就不用插了 在python里，{},[],()，等都等价于False！
                    #     Structure.append(temp_uniprot_structure)
                    # #print('test=', temp_uniprot_structure)
                    # # uniprot_structure = {}
                    if(list2[num]['database'] == "PDB"):
                        structure_temp_dict = copy.deepcopy(list2[num])
                        Structure.append(structure_temp_dict)

                    uniprot_alphafold = {}



                    if (list2[num]['database'] == "AlphaFoldDB"):
                        flag = 1
                        html_url_prediction = "https://alphafold.ebi.ac.uk/api/prediction/" + UPID  # alphafold prediction 接口请求方法也为get 用来获取alphafold
                        # response = request.urlopen(html_url_prediction) #可能会出现网络中断问题，给它加一个判错替换成下面的样子
                        while True:
                            try:
                                response = request.urlopen(html_url_prediction) #请求成功
                                break
                            except Exception as e:
                                # 如果发生异常，则打印异常信息，并等待一段时间后重新尝试请求
                                print("网页请求失败:", e)
                                print("等待一段时间后重新尝试请求...")
                                time.sleep(5)  # 等待5秒后重新尝试请求

                        html = response.read()
                        # print("prediction html = ",html)
                        temp_json = str(html).split("b'")[1].strip("'")
                        #print("prediction html json = ", type(temp_json))
                        try:
                            eval(temp_json)
                        except Exception as error:
                            print('错误类型是', error.__class__.__name__)
                            print('错误明细是', error)
                            eval_error = 1
                        if (eval_error == 0):
                            real_dict1 = eval(temp_json)
                            #print("real dict type = ", type(real_dict1), "real dict = ", real_dict1)

                            uniprot_alphafold["Source"] = "AlphaFoldDB"
                            uniprot_alphafold["IDENTIFIER_entryId"] = real_dict1[0]["entryId"]
                            uniprot_alphafold["gene"] = real_dict1[0]["gene"]
                            uniprot_alphafold["uniprotAccession"] = real_dict1[0]["uniprotAccession"]
                            uniprot_alphafold["METHOD"] = "Predicted"
                            # else:
                            #     uniprot_alphafold["Source"] = "AlphaFoldDB"
                            #     uniprot_alphafold["METHOD"] = "Predicted"
                            #print("kek", uniprot_alphafold)
                            record["Structure_AlphaFold"] = uniprot_alphafold

                        else:
                            uniprot_alphafold["Source"] = "AlphaFoldDB"
                            uniprot_alphafold["METHOD"] = "Predicted"
                            uniprot_alphafold["eval_false"] = "True"
                            record["Structure_AlphaFold"] = uniprot_alphafold
                            print("字符串转字典失败，要用其他方法解析json文件")
                    if(flag == 0):#可能会没有AlphaFold
                        record["Structure_AlphaFold"] = 'null'
                    num += 1

                record["Structure_PDB"] = Structure
            else:
                record["Structure_PDB"] = 'null'

            record["Relate_disease"] = 'null'
            record["Binding_pocket"] = 'null'
            record["Postivecompound"] = 'null'
            record["Remark"] = 'null'
            record["Topology"] = 'null'
            # countid += 1

            filtered_list = jsonpath(real_dict, '$..uniProtKBCrossReferences')
            """
            $..uniProtKBCrossReferences  '$..' 表示递归地查询目标下所有层级
            这意味着 JSONPath 将会在 JSON 结构中深入递归，找到所有名为 "uniProtKBCrossReferences" 的键，并返回它们所对应的值。
            """
            # filtered_list2 = jsonpath(real_dict,"$..database") # $.database[?(@ == 'Go')]
            # print(filtered_list)
            listconcat = (list(_flatten(filtered_list)))
            num = 0
            Go_info_list = []
            while (num <= len(listconcat) - 1):  # 遍历整个filtered_list (里面的元素是字典)
                if (bool(listconcat[num]['database'])):  # 先看有没有database这个字段
                    # if (bool(listconcat[num]['database'][0]) and bool(listconcat[num]['properties'][0]['key'])):  # 再看properties tag下的列表是不是空表，不是的话第一个元素有没有key
                    if (listconcat[num]['database'] == 'GO'):  # 再看它的值是不是 GO
                        Go_info_list.append(listconcat[num])
                num += 1
            record["Go_info"] = Go_info_list



        else:#没办法通过eval() 将json转字典了,那就只能读json文件了,这里先跳过，以后再加
            #real_dict = json.loads(temp_dict)
            No += 1
            countid += 1
            record = {}
            print("skip forward")
            skip_id_list.append(UPID)
            #record["No"] = No  # 自增id NO
            record["UPID"] = UPID
            skip_id_dict["error_upid"] = skip_id_list
            temp_dict = copy.deepcopy(skip_id_dict)
            dev2.insert_one(temp_dict)

        #dev.insert_one(record)


        dev.update_one({'UPID': record['UPID']}, {'$set': record}, True) # 避免重复存取，只存储没有的，或更新现有的


        # result = dev.insert_many(molecule_list)
        # print(len(result.inserted_ids), 'molecule records inserted!')


        # result = dev.updata_one({"UPID": UPID}, {"$set": record})

        #print('records inserted!')
        #records.append(record)

        # result = dev.insert_many(records)#mongodb insert_many
        # print(len(result.inserted_ids), 'records inserted!')


def testscrapy(testUPID):
    html_url = "https://rest.uniprot.org/uniprotkb/" + testUPID
    response = request.urlopen(html_url)
    html = response.read()
    # print("html:", html)
    temp_dict = str(html).split("b'")[1].strip("'")

    global false, null, true
    false = null = true = ''  # 由于eval() 不能处理false，因此要先将false设为全局变量，并将其置空
    real_dict = eval(temp_dict)
    filtered_list = jsonpath(real_dict, '$..uniProtKBCrossReferences')
    """
    $..uniProtKBCrossReferences  '$..' 表示递归地查询目标下所有层级
    这意味着 JSONPath 将会在 JSON 结构中深入递归，找到所有名为 "uniProtKBCrossReferences" 的键，并返回它们所对应的值。
    """
    # filtered_list2 = jsonpath(real_dict,"$..database") # $.database[?(@ == 'Go')]
    #print(filtered_list)
    listconcat = (list(_flatten(filtered_list)))
    num = 0
    Go_info_list = []
    while (num <= len(listconcat) - 1): #遍历整个filtered_list (里面的元素是字典)
        if (bool(listconcat[num]['database'])):  # 先看有没有database这个字段
            # if (bool(listconcat[num]['database'][0]) and bool(listconcat[num]['properties'][0]['key'])):  # 再看properties tag下的列表是不是空表，不是的话第一个元素有没有key
            if (listconcat[num]['database'] == 'GO'):  # 再看它的值是不是 GO
                Go_info_list.append(listconcat[num])
        num += 1
    print(Go_info_list)

if __name__ == '__main__':

    """
    介绍一下爬虫这个脚本
    mainscrapy() 是这个爬虫的主函数，用来爬取除PDB文件之外的所有目标信息（从Uniprot,AlphaFold,PDB三个开源数据库爬取）
    testscrapy() 对某些特殊的蛋白质进行爬虫测试
    getEntryIDFromMongo_ThenDownload() 爬取对应UPID索引到的AlphaFold entryID的预测PDB结构文件
    
    """

    # testUPID = input("输入你要测试的UPID=") #P05067(无论是PDB结构还是AlphaFold预测结构，无论是Go标签还是Ec标签,都非常全!),Q14118(只有一条mf,缺少bp和cc)
    # testscrapy(testUPID)

    """从Uniprot,PDB,AlphaFold 数据库关联爬取一条UPID对应的蛋白质的相关信息存入本地MongoDB"""
    mainscrapy()


    """开始爬取AlphaFold中的PDB文件"""
    # getEntryIDFromMongo_ThenDownload('A0A087X1C5',"G:/fkfk/AlphaFold_refine/biotoolbox/TestPDBDataset/") # UPID , savepath


