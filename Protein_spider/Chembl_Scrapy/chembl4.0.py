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

mongoClient = MongoClient('mongodb://localhost:27017/')
dtmr_dev = mongoClient.get_database('Chembl_DOC')
'''
devhead作为活性物信息表，作为所有表的主表，里面存了
"activity_id" : 活性物id，可通过这个id从 https://www.ebi.ac.uk/chembl/api/data/activity 接口中获取该活性物信息。

"molecule_chembl_id" : 小分子化合物id 从 https://www.ebi.ac.uk/chembl/api/data/molecule 接口中获取该小分子化合物的信息。
"target_chembl_id" : 靶点蛋白id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/target 接口中获取对应的靶点信息。
"assay_chembl_id" : 参考文献 id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/assay 接口中获取对应的文献信息。
"document_chembl_id" : 文档id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/document 接口中获得对应的文档信息。

【注】接口名称ChEMBL ID Lookup，该接口用来得到所有的chembl ID，可以通过修改后面的limit来控制获取的chembl ID 数量，
网页默认都是20个这和chembl网页一张只放20个数据有关，下二十个需要翻页。

接口新发现！  chembl网页有断点请求功能：
https://www.ebi.ac.uk/chembl/api/data/activity?limit=1&offset=2
limit 是网页接口响应后传回的数据条数限制，offset相当于chembl网页的断点 如：
limit = 20
offset = 1
说明接口会从第1条数据开始返回，返回数据条数为20条
['31863', '31864', '31865', '31866', '31867', '31868', '31869', '31870', '31871', '31872', '31873', '31874', '31875', '31876', '31877', '31878', '31879', '31880', '31881', '31882']
若
limit = 10
offset = 10
返回值就为
['31873', '31874', '31875', '31876', '31877', '31878', '31879', '31880', '31881', '31882']

'''
devhead = dtmr_dev.get_collection('Activity')#活性物信息表，作为所有表的主表，里面存了所有的关联id，详情见上方注释

dev = dtmr_dev.get_collection('Compound_molecule')#小分子化合物表，主键为molecule_chembl_id
dev1 = dtmr_dev.get_collection('Assay')#参考文献表，主键为assay_chembl_id
dev2 = dtmr_dev.get_collection('Target')#靶点蛋白表，主键为target_chembl_id
dev3 = dtmr_dev.get_collection('Document')#文档表，主键为document_chembl_id
dev4 = dtmr_dev.get_collection('Drugs')#药物表，主键为drug_code

devlog = dtmr_dev.get_collection('Log')#用来保存当日的log日志下次的断点就设在前一天的start_point offset + limit

'''activities 活性物接口爬取'''

limit=input("你要几条activities id？？")
start_point=input('从第几个数据开始？？')
next_start_point = int(start_point) + int(limit)
excute_time = time.asctime()
log = {'最新设置断点位置':start_point,'最新爬取条数':limit,'该日志创建时间':excute_time,'下次断点设置位置offset = ':next_start_point}
# devlog.insert_one(log)#这个log的插入一般放在循环的最后再插入，因为如果运行中途发生了错误如断网，或者某些不确定因素运行中断时，log将无法插入，这也是一个判断程序是否完整执行的有效手段。

html_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit="+str(limit)+"&offset="+str(start_point)#直接从start_point处开始拿activities  接口给的前limit条activities数据

# html_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit=5"#直接拿它给的前5条activity
response = request.urlopen(html_url)
XML = response.read()
soup = BeautifulSoup(XML,'xml')
all_activities = soup.activities.find_all('activity',recursive=False)#recursive=False，关闭递归，说明只需要第一层activity，不需递归遍历他的子孙节点。
#all_activities 是一个长度为tag为<activity>个数的列表，列表元素类型为 <class 'bs4.element.Tag'> 所以可以对列表中的每个元素进行bs解析。

activities_id_list = []#不需要去重，作为主表的主键他是唯一的
molecule_id_list = []#去重过后的id表，避免重复存取，更新。
target_id_list = []
assay_id_list = []
document_id_list = []

relate_list_molecule = []#完整的没去重，方便后续找一对多，和多对多交叉关联
relate_list_target = []
relate_list_assay = []
relate_list_document = []

i=0
activitise_list = []
while(i <= len(all_activities)-1 ):
    activitise_dict = {}

    ligand={}#作为活性物字典activitise_dict 中的ligand嵌套字典。
    activitise_dict['activity_id'] = all_activities[i].activity_id.string
    activitise_dict['molecule_chembl_id'] = all_activities[i].molecule_chembl_id.string

    if (all_activities[i].activity_id.string not in activities_id_list):  # 其实activity_id 是主表的主键，他是唯一的，不需要这个条件判断。但为了防止万一还是加上了。
        activities_id_list.append(all_activities[i].activity_id.string)

    if (all_activities[i].molecule_chembl_id.string not in molecule_id_list):#两个条件判断，一个去除当次循环中的重复id，一次去除整个mongodb中已有的重复id
        if (dev.count_documents({"molecule_chembl_id": all_activities[i].molecule_chembl_id.string}) == 0):  # 判断这个molecule_chembl_id是否在mongodb里面已经存在了,存在就不需要在插入id list了
            molecule_id_list.append(all_activities[i].molecule_chembl_id.string)  # 一个化合物小分子和某个靶点的活性物是多对多的关系，所以会有重复的id，不需要重复的id
    relate_list_molecule.append(all_activities[i].molecule_chembl_id.string)#保存存了所有id（包括重复id）的副本，之后再对副本进行关联操作

    if (all_activities[i].target_chembl_id.string not in target_id_list):  # 同理，去掉重复id
        if (dev2.count_documents({"target_chembl_id": all_activities[i].target_chembl_id.string}) == 0):  # 说明mongo里面没有这个id，再插入
            target_id_list.append(all_activities[i].target_chembl_id.string)
    relate_list_target.append(all_activities[i].target_chembl_id.string)

    if (all_activities[i].assay_chembl_id.string not in assay_id_list):  # 同理，去掉重复id
        if (dev1.count_documents({"assay_chembl_id": all_activities[i].assay_chembl_id.string}) == 0):
            assay_id_list.append(all_activities[i].assay_chembl_id.string)
    relate_list_assay.append(all_activities[i].assay_chembl_id.string)

    if (all_activities[i].document_chembl_id.string not in document_id_list):  # 同理，去掉重复id
        if (dev3.count_documents({"document_chembl_id": all_activities[i].document_chembl_id.string}) == 0):
            document_id_list.append(all_activities[i].document_chembl_id.string)
    relate_list_document.append(all_activities[i].document_chembl_id.string)


    activitise_dict['parent_molecule_chembl_id'] = all_activities[i].parent_molecule_chembl_id.string
    activitise_dict['target_tax_id'] = all_activities[i].target_tax_id.string
    activitise_dict['record_id'] = all_activities[i].record_id.string

    #compound_key属性在compound 以及molecule接口中都找不到，必须通过record id 访问Compound Record 接口才能找到。
    #这个record id在主表activity内，可以通过它访问Compound Record 接口https://www.ebi.ac.uk/chembl/api/data/compound_record/ +str(record id)
    compound_url = 'https://www.ebi.ac.uk/chembl/api/data/compound_record/'+str(all_activities[i].record_id.string)
    response1 = request.urlopen(compound_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    activitise_dict['Compound_key'] = soup1.compound_record.compound_key.string

    activitise_dict['Standard_type'] = all_activities[i].standard_type.string
    activitise_dict['standard_units'] = all_activities[i].standard_units.string

    activitise_dict['Standard_Relation'] = all_activities[i].standard_relation.string
    activitise_dict['Standard_value'] = all_activities[i].standard_value.string
    activitise_dict['standard_text_value'] = all_activities[i].standard_text_value.string
    activitise_dict['standard_upper_value'] = all_activities[i].standard_upper_value.string
    activitise_dict['units'] = all_activities[i].units.string
    activitise_dict['uo_units'] = all_activities[i].uo_units.string
    activitise_dict['value'] = all_activities[i].uo_units.string
    activitise_dict['upper_value'] = all_activities[i].upper_value.string
    if(bool(all_activities[i].pchembl_value)):#当相关属性可能不存在时，需要先条件判断,避免判空后报错。
        activitise_dict['pChEMBL_Value'] = all_activities[i].pchembl_value.string
    else:
        activitise_dict['pChEMBL_Value'] = 'null'
    if(bool(all_activities[i].activity_comment)):
        activitise_dict['Comment'] = all_activities[i].activity_comment.string
    else:
        activitise_dict['Comment'] = 'null'

    activitise_dict['assay_description'] = all_activities[i].assay_description.string
    activitise_dict['bao_label'] = all_activities[i].bao_label.string
    activitise_dict['bao_endpoint'] = all_activities[i].bao_endpoint.string
    activitise_dict['bao_format'] = all_activities[i].bao_format.string

    #assay_organism 也需要像上面的compound record一样关联查找，这次是通过assay id 去 接口https://www.ebi.ac.uk/chembl/api/data/assay/ 最后查找到assay_organism
    activitise_dict['assay_chembl_id'] = all_activities[i].assay_chembl_id.string
    assay_url = 'https://www.ebi.ac.uk/chembl/api/data/assay/' + str(all_activities[i].assay_chembl_id.string)
    response1 = request.urlopen(assay_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    if(bool(soup1.assay.assay_organism.string)):
        activitise_dict['assay_organism'] = soup1.assay.assay_organism.string
    else:
        activitise_dict['assay_organism'] = 'null'
    activitise_dict['assay_type'] = all_activities[i].assay_type.string

    activitise_dict['target_name'] = all_activities[i].target_pref_name.string
    activitise_dict['target_organism'] = all_activities[i].target_organism.string
    activitise_dict['target_organism'] = all_activities[i].target_organism.string

    #target_type 也需要通过他的target_id去 接口 https://www.ebi.ac.uk/chembl/api/data/target/ 关联查找。
    activitise_dict['target_chembl_id'] = all_activities[i].target_chembl_id.string
    target_url = 'https://www.ebi.ac.uk/chembl/api/data/target/' + str(all_activities[i].target_chembl_id.string)
    response1 = request.urlopen(target_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    if(bool(soup1.target.target_type)):
        activitise_dict['target_type'] = soup1.target.target_type.string
    else:
        activitise_dict['target_type'] = 'null'

    #doc_type 也要通过document接口 https://www.ebi.ac.uk/chembl/api/data/document/ 关联索引
    activitise_dict['document_chembl_id'] = all_activities[i].document_chembl_id.string

    document_url = 'https://www.ebi.ac.uk/chembl/api/data/document/' + str(all_activities[i].document_chembl_id.string)
    response1 = request.urlopen(document_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    if (bool(soup1.document.doc_type)):
        activitise_dict['doc_type'] = soup1.document.doc_type.string
    else:
        activitise_dict['doc_type'] = 'null'

    #source description 则要通过 src_id 来访问接口：https://www.ebi.ac.uk/chembl/api/data/source/
    activitise_dict['src_id'] = all_activities[i].src_id.string
    src_url = 'https://www.ebi.ac.uk/chembl/api/data/source/' + str(all_activities[i].src_id.string)
    response1 = request.urlopen(src_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    if (bool(soup1.source.src_description)):
        activitise_dict['source description'] = soup1.source.src_description.string
    else:
        activitise_dict['source description'] = 'null'

    #该活性物没有cell_chembl_id
    activitise_dict['action_type'] = all_activities[i].action_type.string
    activitise_dict['activity_comment'] = all_activities[i].activity_comment.string
    activitise_dict['activity_properties'] = all_activities[i].activity_properties.string
    activitise_dict['canonical_smiles'] = all_activities[i].canonical_smiles.string
    activitise_dict['data_validity_comment'] = all_activities[i].data_validity_comment.string
    activitise_dict['data_validity_description'] = all_activities[i].data_validity_description.string
    activitise_dict['data_validity_description'] = all_activities[i].data_validity_description.string

    if (all_activities[i].ligand_efficiency.bei != None or all_activities[i].ligand_efficiency.le != None):
        print('ligand output:',all_activities[i].ligand_efficiency.bei)
        ligand['bei'] = all_activities[i].ligand_efficiency.bei.string
        ligand['le'] = all_activities[i].ligand_efficiency.le.string
        ligand['lle'] = all_activities[i].ligand_efficiency.lle.string
        ligand['sei'] = all_activities[i].ligand_efficiency.sei.string
    activitise_dict['ligand_effieiency'] = ligand
    temp_dict = copy.deepcopy(activitise_dict)#保存物理地址，防止地址干扰导致的覆盖问题。
    activitise_list.append(temp_dict)
    i += 1
print('activity id len=',len(activities_id_list),'activities_id_list:',activities_id_list)
print('molecule id len=',len(molecule_id_list),'molecule_id_list:',molecule_id_list)
print('target id len=',len(target_id_list),'target_id_list:',target_id_list)
print('assay id len=',len(assay_id_list),'assay_id_list:',assay_id_list)
print('document id len=',len(document_id_list),'document_id_list:',document_id_list)
#devhead.insert_many(activitise_list)
'''
update的第一个参数是查询条件，即根据这个条件查询已存在（集合collection）中的数据
update的第二个参数是将要插入的值
update的第三个参数upsert，默认为False

update的第三个参数upsert
False（默认）,只更新，不插入
True, 即更新，也插入
可以通过设置第三个参数为True，再设置第一个参数为（每条记录的）唯一值对每条记录进行过滤，匹配到相同的唯一值，进行更新，没有匹配到，则插入。达到插入前进行去重的效果。
即：collection.update({'name': ele['name']}, {'$set': ele}, True)
'''
#下方通过设置update第三个参数为True，再设置第一个参数为（每条记录的）唯一值对每条记录进行过滤，匹配到相同的唯一值，进行更新，没有匹配到，则插入。达到插入前进行去重的效果。
for element in activitise_list:#避免重复存储，只存储没有的，或更新现有的
    temp = devhead.update_one({'activity_id': element['activity_id']}, {'$set': element}, True)


'''找不同id表之间的相关性 '''
print('full activity id len=',len(activities_id_list),'activities_id_list:',activities_id_list)
print('full molecule id len=',len(relate_list_molecule),'molecule_id_list:',relate_list_molecule)
print('full target id len=',len(relate_list_target),'target_id_list:',relate_list_target)
print('full assay id len=',len(relate_list_assay),'assay_id_list:',relate_list_assay)
print('full document id len=',len(relate_list_document),'document_id_list:',relate_list_document)
relative_list=[activities_id_list,relate_list_molecule,relate_list_target,relate_list_assay,relate_list_document]
#将所有的关系放在一起，做成一个二维列表，方便对比和索引

'''relate_dict 里面包含molecule id 和activities id 之间的关联性'''
#flag = 0
relate_dict_molecule = {}
list1 = []  #用来存放没有重复的molecule id 对应的activities_id（一个列表就存唯一一个activities_id）
list2 = []  # 用来存放该重复的molecule id对应的不同activities_id（一个列表存多个activities_id）
j=0
cache = ''
while(j <= len(relate_list_molecule)-1 ):
    flag = 0
    str1 = relative_list[0][j] #遍历二维数组第一行 即activities_id_list
    str2 = relative_list[1][j] #遍历二维数组第二行 即relate_list_molecule
    while (j+1 <= len(relate_list_molecule)-1 and str2 == relative_list[1][j+1]): #此时的molecule id与下一个molecule id重复
        list2.append(str1)
        list2.append(relative_list[0][j+1])
        # print("此时的molecule id：",str2,"他的list2=",list2)
        j += 1
        flag = 1
    #上面这个while循环结束后会得到一个列表存放当前重复的molecule id对应的所有activities_id
    if(flag == 1): #flag=1 说明发生过上面的while循环
        # print("这个时候的molecule id：", str2, "他的list2=", list2)
        relate_dict_molecule[str2] = list2 # 存入字典关键字为当下的molecule id，值为list2内所有的值。
        #print('dict demo:',relate_dict_molecule)
        list2 = []
        j += 1
    elif (j+1 == len(relate_list_molecule)-1):#已经遍历到队尾了尽管此时flag=0 但是它任然是重复的，上面已经插过了不需要再插入了。
        break
    else: #当前molecule id没与下一个发生重复,且没到队尾
        list1.append(str1)
        relate_dict_molecule[str2] = list1
        list1 = []
        j += 1
print("relate_dict_molecule=",relate_dict_molecule)

'''
每个关联关系都像上面一样搞太麻烦了，我这里直接写了两个函数。
act_relative_find(a) 函数
用来寻找所有四个列表
relate_list_molecule = ['CHEMBL113081', 'CHEMBL324340', 'CHEMBL324340', 'CHEMBL109600', 'CHEMBL109600']
relate_list_target = ['CHEMBL1806', 'CHEMBL3921', 'CHEMBL3879801', 'CHEMBL3921', 'CHEMBL3879801']
relate_list_assay = ['CHEMBL663853', 'CHEMBL872937', 'CHEMBL693237', 'CHEMBL872937', 'CHEMBL693238']
relate_list_document = ['CHEMBL1137930', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658']
与活性物列表：
activities_id_list = ['31863', '31864', '31865', '31866', '31867']
的关联关系
传参a是用来确定relative_list[a]
如：用他来确定是要返回化合物表relate_list_molecule和activities_id_list的关系，则a=1 因为 relate_list_molecule属于二维列表relative_list的第二行
第三行 a =2 则是返回 target靶点蛋白表relate_list_target和activities_id_list的关系
最后的返回值是一个关系字典如下 是返回化合物表relate_list_molecule和activities_id_list的关系
{'CHEMBL113081': ['31863'], 'CHEMBL324340': ['31864', '31865'], 'CHEMBL109600': ['31866', '31867']}
keyword 键'CHEMBL324340' 是molecule id 
value 值['31864', '31865'] 是对应molecule id 的活性物id
'''
def act_relative_find(a):
    i = 0
    j = 1
    relate_dict_index = {}
    cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
    position = []
    while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
        position = [relative_list[0][i]]
        j = i + 1
        while (j <= len(relative_list[a]) - 1):
            if (relative_list[a][i] == relative_list[a][j]):
                # position.append(i)
                position.append(relative_list[0][j])
                cache.append(j)
            j += 1
        relate_dict_index[relative_list[a][i]] = position
        i += 1
        while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
            i += 1
    return relate_dict_index
    # for keyword in relate_dict_index:
    #     print('test22222:',keyword)
    #     #relate_dict[keyword] = relate_dict_index[keyword]
    #     for value in relate_dict_index[keyword]:
    #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]
'''
whatever_relative_find(b,a): 作为def act_relative_find(a): 的升级版本
现在的whatever_relative_find(b,a): 
函数可以传两个任意的参数遍历选择 relative_list=[activities_id_list,relate_list_molecule,relate_list_target,relate_list_assay,relate_list_document]
内的任意两行id进行关联比对
如要是想得到relate_list_molecule 化合物 和 relate_list_target 靶点蛋白之间的id关联关系字典 只需要index_dict = whatever_relative_find(1,2) 
即二维列表relative_list 的第二行molecule id行和第三行target id行进行比对
返回的index_dict = {'CHEMBL1806': ['CHEMBL113081'], 'CHEMBL3921': ['CHEMBL324340', 'CHEMBL109600'], 'CHEMBL3879801': ['CHEMBL324340', 'CHEMBL109600']}
keyword 为target id 
value 为不同的target靶点蛋白对应的小分子化合物
'''
def whatever_relative_find(b,a):
    i = 0
    j = 1
    relate_dict_index = {}
    cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
    position = []
    while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
        position = [relative_list[b][i]]
        j = i + 1
        while (j <= len(relative_list[a]) - 1):
            if (relative_list[a][i] == relative_list[a][j]):
                # position.append(i)
                position.append(relative_list[b][j])
                cache.append(j)
            j += 1
        relate_dict_index[relative_list[a][i]] = position
        i += 1
        while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
            i += 1
    return relate_dict_index
    # for keyword in relate_dict_index:
    #     print('test22222:',keyword)
    #     #relate_dict[keyword] = relate_dict_index[keyword]
    #     for value in relate_dict_index[keyword]:
    #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]

print('whatever relative found:',whatever_relative_find(0,1))



'''
Compound_molecule 化合物小分子接口爬取
【注】还需通过上面的relative_find函数和每个化合物小分子对应的target进行关联得到targets信息，再插入molecule_dict中，说明此化合物有活性的对应的靶点，或该靶点对应的活性化合物小分子。
'''

molecule_list = []
count = 0
for i in molecule_id_list:
    print('此时的molecule id=', i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')
    molecule_dict = {}
    molecule_inner_hierarchy = {}
    molecule_inner_properties = {}
    molecule_inner_structures = {}
    molecule_dict['chembl_id'] = i
    molecule_dict['molecule_id'] = i
    if (bool(soup.find('pref_name').string)):
        molecule_dict['name'] = soup.find('pref_name').string
    else:
        molecule_dict['name'] = 'null'
    # print('this molecule have synonym',bool(soup.molecule_synonyms.synonym))

    if (bool(soup.molecule_synonyms.synonym) and bool(soup.molecule_synonyms.synonym.molecule_synonym.string)):
        molecule_dict['synonym'] = soup.molecule_synonyms.synonym.molecule_synonym.string
    else:
        molecule_dict['synonym'] = 'null'
    molecule_dict['type'] = soup.find('molecule_type').get_text()

    if (bool(soup.find('max_phase').string)):
        molecule_dict['Max_phase'] = soup.find('max_phase').string
    else:
        molecule_dict['Max_phase'] = 'null'

    if (bool(soup.molecule_properties.full_mwt.string)):
        molecule_dict['Molecular Weight'] = soup.molecule_properties.full_mwt.string
    else:
        molecule_dict['Molecular Weight'] = 'null'

    # BIOactivities中的某些活性物成分 还要通过activities接口 https://www.ebi.ac.uk/chembl/api/data/activity/ 关联索引
    molecule_dict[
        'active_chembl_id'] = soup.molecule_hierarchy.active_chembl_id.string  # 根据这个activate id可以将此分子化合物和与之有关的活性物相关联,从而得到需要的Bioactivities
    # molecule_dict['Bioactivaties'] = relate_dict_molecule[str(i)]#str(i) 为此时的molecule id,也是relate_dict_molecule 这个关联字典的主键

    # document_url = 'https://www.ebi.ac.uk/chembl/api/data/document/' + str(all_activities[i].document_chembl_id.string)
    # response1 = request.urlopen(document_url)
    # XML1 = response1.read()
    # soup1 = BeautifulSoup(XML1, 'xml')
    # if (bool(soup1.document.doc_type)):
    #     molecule_dict['doc_type'] = soup1.document.doc_type.string
    # else:
    #     molecule_dict['doc_type'] = 'null'

    molecule_dict['Alogp'] = soup.molecule_properties.alogp.string
    molecule_dict['Polar Surface Area'] = soup.molecule_properties.psa.string
    molecule_dict['HBA'] = soup.molecule_properties.hba.string
    molecule_dict['HBD'] = soup.molecule_properties.hbd.string
    molecule_dict['Passes Ro3'] = soup.molecule_properties.ro3_pass.string
    molecule_dict['QED Weighted'] = soup.molecule_properties.qed_weighted.string

    molecule_dict['atc_classifications'] = soup.atc_classifications.string
    molecule_dict['availability_type'] = soup.availability_type.string
    molecule_dict['biotherapeutic'] = soup.biotherapeutic.string
    molecule_dict['black_box_warning'] = soup.black_box_warning.string
    molecule_dict['chemical_probe'] = soup.chemical_probe.string
    molecule_dict['cross_references'] = soup.cross_references.string
    molecule_dict['dosed_ingredient'] = soup.dosed_ingredient.string
    molecule_dict['first_approval'] = soup.first_approval.string
    molecule_dict['first_in_class'] = soup.first_in_class.string
    molecule_dict['helm_notation'] = soup.helm_notation.string
    molecule_dict['indication_class'] = soup.indication_class.string
    molecule_dict['inorganic_flag'] = soup.inorganic_flag.string
    molecule_dict['max_phase'] = soup.max_phase.string
    molecule_dict['chebi_par_id'] = soup.find(
        'chebi_par_id').get_text()  # get_text()和.string作用一致，都是返回对应解析路径下的内容，但当路径下为空时后者会直接报错。
    molecule_dict['chirality'] = soup.find('chirality').get_text()

    molecule_inner_hierarchy['active_chembl_id'] = soup.find('molecule_hierarchy').active_chembl_id.string
    molecule_inner_hierarchy['molecule_chembl_id'] = soup.find('molecule_hierarchy').molecule_chembl_id.string
    molecule_inner_hierarchy['parent_chembl_id'] = soup.find('molecule_hierarchy').parent_chembl_id.string
    molecule_dict['hierarchy'] = molecule_inner_hierarchy

    molecule_inner_properties['alogp'] = soup.find('molecule_properties').alogp.string
    molecule_inner_properties['cx_logd'] = soup.find('molecule_properties').cx_logd.string
    molecule_inner_properties['cx_logp'] = soup.find('molecule_properties').cx_logp.string
    molecule_inner_properties['cx_most_apka'] = soup.find('molecule_properties').cx_most_apka.string
    molecule_inner_properties['cx_most_bpka'] = soup.find('molecule_properties').cx_most_bpka.string
    molecule_inner_properties['full_molformula'] = soup.find('molecule_properties').full_molformula.string
    molecule_inner_properties['full_mwt'] = soup.find('molecule_properties').full_mwt.string
    molecule_inner_properties['hba'] = soup.find('molecule_properties').hba.string
    molecule_inner_properties['hba_lipinski'] = soup.find('molecule_properties').hba_lipinski.string
    molecule_inner_properties['hbd'] = soup.find('molecule_properties').hbd.string
    molecule_inner_properties['hbd_lipinski'] = soup.find('molecule_properties').hbd_lipinski.string
    molecule_inner_properties['heavy_atoms'] = soup.find('molecule_properties').heavy_atoms.string
    molecule_inner_properties['aromatic_rings'] = soup.find('molecule_properties').aromatic_rings.string
    molecule_inner_properties['molecular_species'] = soup.find('molecule_properties').molecular_species.string
    molecule_inner_properties['mw_freebase'] = soup.find('molecule_properties').mw_freebase.string
    molecule_inner_properties['mw_monoisotopic'] = soup.find('molecule_properties').mw_monoisotopic.string
    molecule_inner_properties['np_likeness_score'] = soup.find('molecule_properties').np_likeness_score.string
    molecule_inner_properties['num_lipinski_ro5_violations'] = soup.find(
        'molecule_properties').num_lipinski_ro5_violations.string
    molecule_inner_properties['num_ro5_violations'] = soup.find('molecule_properties').num_ro5_violations.string
    molecule_inner_properties['psa'] = soup.find('molecule_properties').psa.string
    molecule_inner_properties['qed_weighted'] = soup.find('molecule_properties').qed_weighted.string
    molecule_inner_properties['ro3_pass'] = soup.find('molecule_properties').ro3_pass.string
    molecule_inner_properties['rtb'] = soup.find('molecule_properties').rtb.string
    molecule_dict['properties'] = molecule_inner_properties

    molecule_inner_structures['canonical_smiles'] = soup.find('molecule_structures').canonical_smiles.string
    molecule_inner_structures['molfile'] = soup.find('molecule_structures').molfile.string
    molecule_inner_structures['standard_inchi'] = soup.find('molecule_structures').standard_inchi.string
    molecule_inner_structures['standard_inchi_key'] = soup.find('molecule_structures').standard_inchi_key.string
    molecule_dict['structures'] = molecule_inner_structures

    molecule_dict['natural_product'] = soup.natural_product.string
    molecule_dict['oral'] = soup.oral.string
    molecule_dict['parenteral'] = soup.parenteral.string
    molecule_dict['polymer_flag'] = soup.polymer_flag.string
    molecule_dict['pref_name'] = soup.pref_name.string
    molecule_dict['prodrug'] = soup.prodrug.string
    molecule_dict['structure_type'] = soup.structure_type.string
    molecule_dict['therapeutic_flag'] = soup.therapeutic_flag.string
    molecule_dict['topical'] = soup.topical.string
    molecule_dict['usan_stem'] = soup.usan_stem.string
    molecule_dict['usan_stem_definition'] = soup.usan_stem_definition.string
    molecule_dict['usan_substem'] = soup.usan_substem.string
    molecule_dict['usan_year'] = soup.usan_year.string
    molecule_dict['withdrawn_flag'] = soup.withdrawn_flag.string

    temp_dict = copy.deepcopy(molecule_dict)
    molecule_list.append(temp_dict)
    count += 1

for element in molecule_list:  # 避免重复存取，只存储没有的，或更新现有的
    temp = dev.update_one({'molecule_id': element['molecule_id']}, {'$set': element}, True)
    # result = dev.insert_many(molecule_list)
    # print(len(result.inserted_ids), 'molecule records inserted!')

'''assay 论文，参考文献接口爬取'''
count = 0
assay_list = []
for i in assay_id_list:
    print('Assay count:', count, "id:", i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/assay/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')
    assay_dict = {}
    assay_dict['assay_chembl_id'] = i
    assay_dict['document_chembl_id'] = soup.find('assay').document_chembl_id.string
    assay_dict['target_chembl_id'] = soup.find('assay').target_chembl_id.string
    assay_dict['cell_chembl_id'] = soup.find('assay').cell_chembl_id.string
    assay_dict['tissue_chembl_id'] = soup.find('assay').tissue_chembl_id.string
    assay_dict['variant_sequence'] = soup.find('assay').variant_sequence.string

    assay_dict['assay_organism'] = soup.find('assay').assay_organism.string
    assay_dict['assay_parameters'] = soup.find('assay').assay_parameters.string
    assay_dict['assay_strain'] = soup.find('assay').assay_strain.string
    assay_dict['assay_subcellular_fraction'] = soup.find('assay').assay_subcellular_fraction.string

    assay_dict['assay_tax_id'] = soup.find('assay').assay_tax_id.string
    assay_dict['assay_type'] = soup.find('assay').assay_type.string
    assay_dict['assay_type_description'] = soup.find('assay').assay_type_description.string
    assay_dict['bao_format'] = soup.find('assay').bao_format.string
    assay_dict['bao_label'] = soup.find('assay').bao_label.string
    assay_dict['confidence_description'] = soup.find('assay').confidence_description.string
    assay_dict['confidence_score'] = soup.find('assay').confidence_score.string
    assay_dict['description'] = soup.find('assay').description.string
    # assay_dict['document_chembl_id'] = soup.find('assay').document_chembl_id.string
    # assay_dict['target_chembl_id'] = soup.find('assay').target_chembl_id.string
    assay_dict['relationship_description'] = soup.find('assay').relationship_description.string
    assay_dict['relationship_type'] = soup.find('assay').relationship_type.string

    # source description 要通过 src_id 来访问接口：https://www.ebi.ac.uk/chembl/api/data/source/ 关联查询
    assay_dict['src_id'] = soup.assay.src_id.string
    src_url = 'https://www.ebi.ac.uk/chembl/api/data/source/' + str(soup.assay.src_id.string)
    response1 = request.urlopen(src_url)
    XML1 = response1.read()
    soup1 = BeautifulSoup(XML1, 'xml')
    if (bool(soup1.source.src_description)):
        assay_dict['source description'] = soup1.source.src_description.string
    else:
        assay_dict['source description'] = 'null'

    temp_dict = copy.deepcopy(assay_dict)
    assay_list.append(temp_dict)
    count += 1

for element in assay_list:#避免重复存取，只存储没有的，或更新现有的
    temp = dev1.update_one({'assay_chembl_id': element['assay_chembl_id']}, {'$set': element}, True)
# dev1.insert_many(assay_list)


'''target 靶点蛋白接口爬取'''
target_list = []
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
    # print('accession error find:',bool(soup.target.target_components.target_component))
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
    if (bool(soup.target.target_components.target_component)):
        all_synonums = soup.target.target_components.target_component.target_component_synonyms.find_all(
            'target_component_synonym', recursive=False)
    else:
        all_synonums = []
        target_dict['target_component_synonyms'] = 'null'
    if (bool(soup.target.target_components.target_component)):
        all_xrefs = soup.target.target_components.target_component.target_component_xrefs.find_all('target',
                                                                                                   recursive=False)
    else:
        all_xrefs = []
        target_dict['target_component_xrefs'] = 'null'

    i = 0
    if (len(all_synonums) >= 1):
        while (i <= len(all_synonums) - 1):
            component_synonym.append(all_synonums[i].component_synonym.string)
            syn_type.append(all_synonums[i].syn_type.string)
            i += 1
        target_component_synonyms['component_synonym'] = component_synonym
        target_component_synonyms['syn_type'] = syn_type
        target_dict['target_component_synonyms'] = target_component_synonyms
    if (len(all_xrefs) >= 1):
        i = 0
        xref_src_db_list = []
        while (i <= len(all_xrefs) - 1):
            if (all_xrefs[i].xref_src_db.string not in xref_src_db_list):
                xref_src_db_list.append(all_xrefs[i].xref_src_db.string)
            i += 1
        for j in xref_src_db_list:
            data_source = str(j)
            j = {}  # 不同的数据库来源类型作不同的字典
            num = 0
            xref_id = []
            xref_name = []
            while (num <= len(all_xrefs) - 1):  # 循环结束会得到j（src name）对应的两个列表xref id表和xref name表
                if (all_xrefs[num].xref_src_db.string == data_source):
                    xref_id.append(all_xrefs[num].xref_id.string)
                    xref_name.append(all_xrefs[num].xref_name.string)
                num += 1
            j['xref_id'] = xref_id
            j['xref_name'] = xref_name
            temp_dict = copy.deepcopy(j)
            target_dict[data_source] = temp_dict

    '''该target 和molecule 关联'''
    index_dict = whatever_relative_find(1,2)#用到的该函数详情见relative_found.py注释
    target_dict['Compounds'] = index_dict[soup.target.target_chembl_id.string]

    '''该target和activities关联'''
    index_dict=act_relative_find(2)#用到的该函数详情见relative_found.py注释
    target_dict['Activities'] = index_dict[soup.target.target_chembl_id.string]

    target_dict['target_type'] = soup.target.target_type.string
    target_dict['tax_id'] = soup.target.tax_id.string

    # html_url = "https://www.ebi.ac.uk/chembl/api/data/target?limit=5"  # 直接拿它给的前5条target
    # response = request.urlopen(html_url)
    # XML = response.read()
    # soup = BeautifulSoup(XML, 'xml')
    # all_target = soup.targets.find_all('target', recursive=False)
    # all_target 是一个长度为五的列表，列表元素类型为 <class 'bs4.element.Tag'> 所以可以对列表中的每个元素进行bs解析。
    # print(all_target[0].target_type.string)   =SINGLE PROTEIN
    # dict1 = {}  # 用来当做crossreferences 里面的内嵌字典target
    # dict1['xref_id'] = all_target[i].cross_references.find_all('target', recursive=False)[0].xref_id.string
    # dict1['xref_src'] = all_target[i].cross_references.find_all('target', recursive=False)[0].xref_src.string
    # target_inner_crossreference['target1'] = dict1
    # target_dict['cross_reference'] = target_inner_crossreference
    # dict2 = {}  # 用来当作target_components 里面的内嵌字典target_component
    # dict2['accession'] = all_target[i].target_components.target_component.accession.string
    # dict2['component_description'] = all_target[i].target_components.target_component.component_description.string
    # dict2['component_id'] = all_target[i].target_components.target_component.component_id.string
    # dict2['component_type'] = all_target[i].target_components.target_component.component_type.string
    # dict2['relationship'] = all_target[i].target_components.target_component.relationship.string
    # target_dict['target_components'] = dict2

    temp_dict = copy.deepcopy(target_dict)
    target_list.append(temp_dict)
    count += 1

for element in target_list:  # 避免重复存取，只存储没有的，或更新现有的
    temp = dev2.update_one({'target_chembl_id': element['target_chembl_id']}, {'$set': element}, True)
# dev2.insert_many(target_list)

'''document 相关文档接口爬取'''
document_list = []
count = 0

for i in document_id_list:
    print('此时的document id=', i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/document/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')
    document_dict = {}
    chembl_release = {} #document_dict的内嵌字典，说明该文献在chembl数据库收录的信息
    chembl_release_list = []
    document_dict['document_chembl_id'] = str(i)
    document_dict['abstract'] = soup.document.abstract.string
    document_dict['authors'] = soup.document.authors.string
    chembl_release['chembl_release'] = soup.document.chembl_release.chembl_release.string
    chembl_release['creation_date'] = soup.document.chembl_release.creation_date.string
    chembl_release_list.append(chembl_release)
    document_dict['chembl_release'] = chembl_release_list
    document_dict['doc_type'] = soup.document.doc_type.string
    document_dict['doi'] = soup.document.doi.string
    document_dict['first_page'] = soup.document.first_page.string
    document_dict['last_page'] = soup.document.last_page.string
    document_dict['issue'] = soup.document.issue.string
    document_dict['journal'] = soup.document.journal.string
    document_dict['journal_full_title'] = soup.document.journal_full_title.string
    document_dict['pubmed_id'] = soup.document.pubmed_id.string
    document_dict['title'] = soup.document.title.string
    document_dict['volume'] = soup.document.volume.string
    document_dict['year'] = soup.document.year.string
    temp_dict = copy.deepcopy(document_dict)
    document_list.append(temp_dict)
    count += 1

for element in document_list:#避免重复存储，只存储没有的，或更新现有的
    temp = dev3.update_one({'document_chembl_id': element['document_chembl_id']}, {'$set': element}, True)
# dev3.insert_many(document_list)


'''drug 关联药物接口爬取'''
html_url = "https://www.ebi.ac.uk/chembl/api/data/drug?limit=1"#直接拿它给的第一条drug
response = request.urlopen(html_url)
XML = response.read()
soup = BeautifulSoup(XML,'xml')
all_drug = soup.drugs.find_all('drug',recursive=False)

i=0
drug_list = []
while(i <= len(all_drug)-1 ):
    drug_dict = {}
    applicants = {} #作为 drug_dict的内嵌字典
    molecule_properties = {}
    molecule_structures = {}
    molecule_synonyms = {}
    reserch_codes = {} #作为 drug_dict的内嵌字典
    synonyms = {} #作为 drug_dict的内嵌字典
    #print(i,'dont have drug error?:',bool(all_drug[i].atc_classification and all_drug[i].atc_classification.drug.code.string))
    if(bool(all_drug[i].atc_classification and all_drug[i].atc_classification.drug.code.string)):
        #没有的就要判个错,并将其置空,这个例子的第五个drug由于连atc_classification这个tag都没有，所以还要一起判断这个tag是否存在否则会直接报错。
        #print('error find:',bool(all_drug[i].atc_classification.drug.code.string))
        drug_dict['drug_code'] = all_drug[i].atc_classification.drug.code.string
        drug_dict['drug_description'] = all_drug[i].atc_classification.drug.description.string
    else:
        drug_dict['drug_code'] = 'null'
        drug_dict['drug_description'] = 'null'

    drug_dict['development_phase'] = all_drug[i].development_phase.string
    drug_dict['first_approval'] = all_drug[i].first_approval.string
    drug_dict['indication_class'] = all_drug[i].indication_class.string
    drug_dict['molecule_chembl_id'] = all_drug[i].molecule_chembl_id.string
    drug_dict['ob_patent'] = all_drug[i].ob_patent.string
    drug_dict['sc_patent'] = all_drug[i].sc_patent.string
    drug_dict['usan_year'] = all_drug[i].usan_year.string
    drug_dict['usan_stem'] = all_drug[i].usan_stem.string
    drug_dict['usan_stem_definition'] = all_drug[i].usan_year.string
    drug_dict['usan_stem_substem'] = all_drug[i].usan_stem_substem.string
    drug_dict['withdrawn_year'] = all_drug[i].withdrawn_year.string
    applicants_value = []
    all_value = all_drug[i].applicants.find_all('value')
    j = 0
    while (j <= len(all_value)-1):
        applicants_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(applicants_value)
    applicants['value'] = temp_list
    drug_dict['applicants'] = applicants

    if(bool(all_drug[i].molecule_properties)):
        molecule_properties['alogp'] = all_drug[i].molecule_properties.alogp.string
        molecule_properties['aromatic_rings'] = all_drug[i].molecule_properties.aromatic_rings.string
        molecule_properties['cx_logd'] = all_drug[i].molecule_properties.cx_logd.string
        molecule_properties['cx_logp'] = all_drug[i].molecule_properties.cx_logp.string
        molecule_properties['cx_most_apka'] = all_drug[i].molecule_properties.cx_most_apka.string
        molecule_properties['cx_most_bpka'] = all_drug[i].molecule_properties.cx_most_bpka.string
        molecule_properties['full_molformula'] = all_drug[i].molecule_properties.full_molformula.string
        molecule_properties['full_mwt'] = all_drug[i].molecule_properties.full_mwt.string
        molecule_properties['molecular_species'] = all_drug[i].molecule_properties.molecular_species.string
        molecule_properties['mw_freebase'] = all_drug[i].molecule_properties.mw_freebase.string
        molecule_properties['mw_monoisotopic'] = all_drug[i].molecule_properties.mw_monoisotopic.string
        molecule_properties['np_likeness_score'] = all_drug[i].molecule_properties.np_likeness_score.string
        molecule_properties['psa'] = all_drug[i].molecule_properties.psa.string
        molecule_properties['qed_weighted'] = all_drug[i].molecule_properties.qed_weighted.string
        molecule_properties['ro3_pass'] = all_drug[i].molecule_properties.ro3_pass.string
        molecule_properties['rtb'] = all_drug[i].molecule_properties.rtb.string
        drug_dict['molecule_properties'] = molecule_properties
    else:
        drug_dict['molecule_properties'] = 'null'

    if (bool(all_drug[i].molecule_structures)):
        molecule_structures['canonical_smiles'] = all_drug[i].molecule_structures.canonical_smiles.string
        molecule_structures['molfile'] = all_drug[i].molecule_structures.molfile.string
        molecule_structures['standard_inchi'] = all_drug[i].molecule_structures.standard_inchi.string
        molecule_structures['standard_inchi_key'] = all_drug[i].molecule_structures.standard_inchi_key.string
        drug_dict['molecule_structures'] = molecule_structures
    else:
        drug_dict['molecule_structures'] = 'null'

    if(bool(all_drug[i].molecule_synonyms)):
        drugs = []
        all_drugmolecule_synonyms = all_drug[i].molecule_synonyms.find_all('drug',recursive=False)
        count_no = 0
        while(count_no <= len(all_drugmolecule_synonyms)-1):
            drugs_dict = {}#注意与之前的drug_dict作区分，这个drugs_dict是作为molecule_synonyms化合物别名字典的内嵌字典
            drugs_dict['molecule_synonym'] = all_drugmolecule_synonyms[count_no].molecule_synonym.string
            drugs_dict['syn_type'] = all_drugmolecule_synonyms[count_no].syn_type.string
            drugs_dict['synonyms'] = all_drugmolecule_synonyms[count_no].synonyms.string
            temp_dict = copy.deepcopy(drugs_dict)
            drugs.append(temp_dict)
            count_no += 1

        molecule_synonyms['drugs'] = drugs
    else:
        molecule_synonyms['drugs'] = 'null'

    drug_dict['molecule_synonyms'] = molecule_synonyms


    research_codes_value = []
    all_value = all_drug[i].research_codes.find_all('value')
    j = 0
    while(j <= len(all_value)-1):
        research_codes_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(research_codes_value)
    reserch_codes['value'] = temp_list
    drug_dict['reserch_codes'] = reserch_codes

    synonyms_value = []
    all_value = all_drug[i].find_all('synonyms',recursive=False)[0].find_all('value')#这里把递归参数设为false因为子表中也有synonyms，因此不需要递归查询子表
    #print('test',bool(all_value[1].string))
    j = 0
    while (j <= len(all_value) - 1 and bool(all_value[j].string) == True):
        synonyms_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(synonyms_value)
    synonyms['value'] = temp_list
    drug_dict['synonyms'] = synonyms

    temp_dict = copy.deepcopy(drug_dict)
    drug_list.append(temp_dict)
    i += 1

for element in drug_list:  # 避免重复存取，只存储没有的，或更新现有的
    temp = dev4.update_one({'drug_code': element['drug_code']}, {'$set': element}, True)
# dev4.insert_many(drug_list)

devlog.insert_one(log)#这个log的插入一般放在循环的最后再插入，因为如果运行中途发生了错误如断网，或者某些不确定因素运行中断时，log将无法插入，这也是一个判断程序是否完整执行的有效手段。



'''
1.chembl 项目总结：
1.1 ChEMBL数据库介绍：
ChEMBL是一个大型的、开放访问的药物发现数据库，旨在收集药物研究和开发过程中的药物化学数据和知识。
有关小分子及其生物活性的信息来自几种核心药物化学期刊的全文文章，并与已批准的药物和临床开发候选药物的数据(如作用机制和治疗适应症)相结合。
生物活性数据还与其他数据库的数据进行交换，如PubChem BioAssay和BindingDB。
1.2 项目流程：（以chembl4.0.py为主要代码）
1.2.1 第一部分：对数据库data接口的请求
devhead作为活性物信息表，作为所有表的主表，里面存了
"activity_id" : 活性物id，可通过这个id从 https://www.ebi.ac.uk/chembl/api/data/activity 接口中获取该活性物信息。

"molecule_chembl_id" : 小分子化合物id 从 https://www.ebi.ac.uk/chembl/api/data/molecule 接口中获取该小分子化合物的信息。
"target_chembl_id" : 靶点蛋白id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/target 接口中获取对应的靶点信息。
"assay_chembl_id" : 参考文献 id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/assay 接口中获取对应的文献信息。
"document_chembl_id" : 文档id 通过这个id从 https://www.ebi.ac.uk/chembl/api/data/document 接口中获得对应的文档信息。
还有drug.......等等接口

【注】接口名称ChEMBL ID Lookup，该接口用来得到所有的chembl ID，可以通过修改后面的limit来控制获取的chembl ID 数量，
网页默认都是20个这和chembl网页一张只放20个数据有关，下二十个需要翻页。

对上述接口验证后可知，返回的 XML 的内容就是上述对应接口的内容，且是一个XML网页文件不再是JSON了无法用JSONpath遍历它，想要遍历他或许要用beautifulsoup进行解析XML?
且可以看到改XML文件的末尾是有一个<page_meta>的板块内<limit>20</limit>，<total_count>4547276</total_count>说明一共有4547276个chembl ID但这里设置最多返回20个
那我就先拿20个呗 =_=
【注】这20条ID里面不全是化合物小分子，还有Assay,必须将他们区分开，否则之后接口会调用报错。

之后我发现这个20条的限制是可以自己修改的比如
随便进去一个接口看他的xml文档可以发现，<page_meta>这个tag下面的<next>里面设置了limit为20这也是应为Chembl网页的一页就存放20个数据，再想要跟多的数据就得翻页
而在Xml里面我们知道了如何修改获得更多的数据如下：
https://www.ebi.ac.uk/chembl/api/data/target + 后缀： ?limit=100&offset=200 这样的话limit就会变成100 返回100条数据。

接口新发现！  chembl网页有断点请求功能：
https://www.ebi.ac.uk/chembl/api/data/activity?limit=1&offset=2
limit 是网页接口响应后传回的数据条数限制，offset相当于chembl网页的断点 如：
limit = 20
offset = 1
说明接口会从第1条数据开始返回，返回数据条数为20条

1.2.2. 以activities表作为主表，active_id作为主键，创建主表activate内含交叉引用的属性

1.2.3. 通过代码细分每一个id并将其存入不同的列表    molecule_id_list,target_id_list,assay_id_list.......
提高代码运行速率，精简id表：
    if (all_activities[i].molecule_chembl_id.string not in molecule_id_list):#两个条件判断，一个去除当此循环重复id，一次去除整个mongodb重复id
        if (dev.count_documents({"molecule_chembl_id": all_activities[i].molecule_chembl_id.string}) == 0):  # 判断这个molecule_chembl_id是否在mongodb里面已经存在了,存在就不需要在插入id list了
            molecule_id_list.append(all_activities[i].molecule_chembl_id.string)  # 一个化合物小分子和某个靶点的活性物是多对多的关系，所以会有重复的id，不需要重复的id
        # relate_list_molecule.append(all_activities[i].molecule_chembl_id.string)#保存存了所有id（包括重复id）的副本，之后再对副本进行关联操作

1.2.4. 找不同id表之间的关联性 一对一关联以及一对多关联;
详情见代码 relative_found.py

1.2.5. 通过不同的接口对某些表项作交叉引用：
如 :
compound_key属性在compound 以及molecule接口中都找不到，必须通过record id 访问Compound Record 接口才能找到。
这个record id再主表activity内，可以通过它访问Compound Record 接口https://www.ebi.ac.uk/chembl/api/data/compound_record/ record id

1.2.6. 为了使代码适应爬取更大的数据，对其进行进一步精炼（阉割） -> chembl_refine.py

===============================================================================
2. chembl_refine.py 使用手册：
2.1 运行该程序后会让使用者输入两个参数：
2.1.1 你每轮循环需要几条activities id？？(推荐条数50条)大过50条的话系统缓存可能不够容易崩溃，你电脑好就没话说但也不易过大，毕竟中途如果断网那也会浪费很多时间。
这个参数说明你每次循环会爬取多少条activate活性物数据以及这些活性物数据主表对应的其他几个次表。
2.1.2 新的断点从第几条数据开始？？
这个参数是设置断点的参数 即offset，断点为几，代码将向chembl data接口发送请求从第几条数据开始返回。

2.2 log日志介绍：
上面两个参数怎么设置都可以从log表中获得，log是一个与这些表处于同源collection路径下的一个表。
每次运行成功一次（所有循环结束）就会生成此次运行的一个log日志，下次运行的参数选择就参照最近（日志有时间属性）的一次日志就行。

2.3 第五十七行 while(int(start_point) < 5000):#循环结束将抓取前 5000 条的activities内容
如果前5000条都存完了，这个5000就手动设置成你接下来要的条数就行，如设置成6000 ，代码就会从startpoint=5000 的地方开始运行，直到startpoint >= 6000 退出循环，此时mongodb中新增5001~6000的数据。

2.4 由于chembl_refine.py为阉割版所以删减了很多内容，以及功能如：不能找activate与target，molecule等的一对多和多对多关系，如果后续想要再添加这重关系可以参考代码chembl4.0.py

2.5 如果程序运行过程中发生断电、断网等不确定因素，即整个循环并没有结束就被强行打断，则代码不会生成最后的log，同理没有log说明运行过程中发生错误，但这并不说明断网/断电之前的那些数据也没了。
这些数据是已经存进数据库里面了的，下一次的断点就需要参照activate表的最后一个active id是第几个来确定。

2.6 由于该项目需要调用的接口太多了，所以速度很慢是正常的，消耗时间也和网速以及对象服务器的稳定性有关（chembl网站服务器）由于这是一个境外网站，数据接口并不是很稳定，经常会请求失败。
后续可以给代码加上判错，请求失败就让他一直请求下去，直至成功。

'''