
#该代码用于将 drug_indication，drug_mechanism，drug_warning，以及metabolism 通过record_id四合一.


import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log


# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')

devhead = dtmr_dev.get_collection('Chembl_Ultra_Drugs')
devlog = dtmr_dev.get_collection('Chembl_MySQL_Log')


connection_success = 0
while(connection_success== 0):
    try:
        db = pymysql.connect(host='172.10.10.8', user='root', passwd='Qaz123456', port=3306, database='chembl_32')#charset='utf-8',
        cursor = db.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor
        cursor.execute("SELECT VERSION()")  # 使用 execute()  方法执行 SQL 查询
        data = cursor.fetchone()  # 使用fetchone()方法获取单条数据
        print('连接成功！',"Database version = %s" % data)
        connection_success = 1
    except:
        print('something wrong! database connection failed')


'''
遍历drug_indication(相对来说数据条数最多)
{"record_id":[{drugind_id_1:{drug_indication_value}},{drugind_id_2:{drug_indication_value}}...........]} 每个record_id可能对应零到多个drugind_id
{"record_id":[{drug_indication_value1},{drug_indication_value2}...........]}
...下一个非重复的record_id
最后再用一个大的列表 drug_indication_summary = [] 汇总
'''



sql_indication = "select record_id from drug_indication limit 0,500"
cursor.execute(sql_indication)
relist = cursor.fetchall() #获取所有的record_id列表
#接下来去重
relist_no_repeat_indication = []
count = 0
while(count <= len(relist)-1):
    if(relist[count] not in relist_no_repeat_indication):
        relist_no_repeat_indication.append(relist[count])
    count += 1

print("relist value =",relist,"lentgh=",len(relist))
print("relist_no_repeat value =",relist_no_repeat_indication,"lentgh=",len(relist_no_repeat_indication))

# drug_indication = []
# record_dict = {}
# drug_indication_summary = []#总表 ，放所有的recordId对应的drug_indication
drug_indication = {}
for record in relist_no_repeat_indication:
    record_id = record[0]
    sql = "select * from drug_indication where record_id = "+"'"+str(record_id)+"'"
    cursor.execute(sql)
    relist2 = cursor.fetchall()#同一record_id下可以有零个到多个drug_indication
    #print("len(relist2)=",len(relist2))
    i = 0

    #{"record_id":[{drugind_id_1:{drug_indication_value}},{drugind_id_2:{drug_indication_value}}...........]} 每个record_id可能对应零到多个drugind_id

    while(i <= len(relist2)-1):
        if relist2[i][1] not in drug_indication:
            drug_indication[relist2[i][1]] = [{"drugind_id":relist2[i][0],"record_id":relist2[i][1],"molregno":relist2[i][2],
                                          "max_phase_for_ind":str(relist2[i][3]),"mesh_id":relist2[i][4],"mesh_heading":relist2[i][5],
                                          "efo_id":relist2[i][6],"efo_term":relist2[i][7]}]
        else:
            drug_indication[relist2[i][1]].append({"drugind_id":relist2[i][0],"record_id":relist2[i][1],"molregno":relist2[i][2],
                                          "max_phase_for_ind":str(relist2[i][3]),"mesh_id":relist2[i][4],"mesh_heading":relist2[i][5],
                                          "efo_id":relist2[i][6],"efo_term":relist2[i][7]})
        i += 1


print("===========================================")


'''
遍历 drug_mechanism
'''

sql_mechanism = "select record_id from drug_mechanism limit 0,500"
cursor.execute(sql_mechanism)
relist = cursor.fetchall() #获取所有的record_id列表
#接下来去重
relist_no_repeat_mechanism = []
count = 0
while(count <= len(relist)-1):
    if(relist[count] not in relist_no_repeat_mechanism):
        relist_no_repeat_mechanism.append(relist[count])
    count += 1

print("relist value =",relist,"lentgh=",len(relist))
print("relist_no_repeat value =",relist_no_repeat_mechanism,"lentgh=",len(relist_no_repeat_mechanism))

drug_mechanism = {}
# record_dict = {}
# drug_mechanism_summary = []#总表 ，放所有的recordId对应的drug_mechanism

for record in relist_no_repeat_mechanism:
    record_id = record[0]
    sql = "select * from drug_mechanism where record_id = " + "'" + str(record_id) + "'"
    cursor.execute(sql)
    relist2 = cursor.fetchall()  # 同一record_id下可以有零个到多个drug_mechanism
    # print("len(relist2)=",len(relist2))
    i = 0


    while (i <= len(relist2) - 1):
        if relist2[i][1] not in drug_mechanism:
            drug_mechanism[relist2[i][1]] = [
                {"mec_id": relist2[i][0], "record_id": relist2[i][1], "molregno": relist2[i][2],
                 "mechanism_of_action": relist2[i][3], "tid": relist2[i][4], "site_id": relist2[i][5],
                 "action_type": relist2[i][6], "direct_interaction": relist2[i][7], "molecular_mechanism":relist2[i][8], "disease_efficacy":relist2[i][9]
                 ,"mechanism_comment":relist2[i][10], "selectivity_comment":relist2[i][11], "binding_site_comment":relist2[i][12], "variant_id":relist2[i][13]
                 }]

            '''
        mechanism["mec_id"] = relist2[i][0]
        mechanism["record_id"] = relist2[i][1]
        mechanism["molregno"] = relist2[i][2]
        mechanism["mechanism_of_action"] = relist2[i][3]
        mechanism["tid"] = relist2[i][4]
        mechanism["site_id"] = relist2[i][5]
        mechanism["action_type"] = relist2[i][6]
        mechanism["direct_interaction"] = relist2[i][7]
        mechanism["molecular_mechanism"] = relist2[i][8]
        mechanism["disease_efficacy"] = relist2[i][9]
        mechanism["mechanism_comment"] = relist2[i][10]
        mechanism["selectivity_comment"] = relist2[i][11]
        mechanism["binding_site_comment"] = relist2[i][12]
        mechanism["variant_id"] = relist2[i][13]
            '''
        else:
            drug_mechanism[relist2[i][1]].append(
                {"mec_id": relist2[i][0], "record_id": relist2[i][1], "molregno": relist2[i][2],
                 "mechanism_of_action": relist2[i][3], "tid": relist2[i][4], "site_id": relist2[i][5],
                 "action_type": relist2[i][6], "direct_interaction": relist2[i][7],
                 "molecular_mechanism": relist2[i][8], "disease_efficacy": relist2[i][9]
                    , "mechanism_comment": relist2[i][10], "selectivity_comment": relist2[i][11],
                 "binding_site_comment": relist2[i][12], "variant_id": relist2[i][13]
                 })
        i += 1


print("===========================================")


'''
遍历 drug_warning
'''

sql_warning = "select record_id from drug_warning limit 0,500"
cursor.execute(sql_warning)
relist = cursor.fetchall() #获取所有的record_id列表
#接下来去重
relist_no_repeat_warning = []
count = 0
while(count <= len(relist)-1):
    if(relist[count] not in relist_no_repeat_warning):
        relist_no_repeat_warning.append(relist[count])
    count += 1

print("relist value =",relist,"lentgh=",len(relist))
print("relist_no_repeat value =",relist_no_repeat_warning,"lentgh=",len(relist_no_repeat_warning))

drug_warning = {}
# record_dict = {}
# drug_warning_summary = []#总表 ，放所有的recordId对应的drug_warning

for record in relist_no_repeat_warning:
    record_id = record[0]
    sql = "select * from drug_warning where record_id = " + "'" + str(record_id) + "'"
    cursor.execute(sql)
    relist2 = cursor.fetchall()  # 同一record_id下可以有零个到多个drug_mechanism
    # print("len(relist2)=",len(relist2))
    i = 0

    # {"record_id":[{drugind_id_1:{drug_indication_value}},{drugind_id_2:{drug_indication_value}}...........]} 每个record_id可能对应零到多个drugind_id

    while (i <= len(relist2) - 1):

        '''
        warning = {}
        warning["warning_id"] = relist2[i][0]
        warning["record_id"] = relist2[i][1]
        warning["molregno"] = relist2[i][2]
        warning["warning_type"] = relist2[i][3]
        warning["warning_class"] = relist2[i][4]
        warning["warning_description"] = relist2[i][5]
        warning["warning_country"] = relist2[i][6]
        warning["warning_year"] = relist2[i][7]
        warning["efo_term"] = relist2[i][8]
        warning["efo_id"] = relist2[i][9]
        warning["efo_id_for_warning_class"] = relist2[i][10]
        '''
        if relist2[i][1] not in drug_warning:
            drug_warning[relist2[i][1]] = [
                {"warning_id": relist2[i][0], "record_id": relist2[i][1], "molregno": relist2[i][2],
                 "warning_type": relist2[i][3], "warning_class": relist2[i][4], "warning_description": relist2[i][5],
                 "warning_country": relist2[i][6], "warning_year": relist2[i][7], "efo_term":relist2[i][8], "efo_id":relist2[i][9]
                 ,"efo_id_for_warning_class":relist2[i][10]}]
        else:
            drug_warning[relist2[i][1]].append(
                {"warning_id": relist2[i][0], "record_id": relist2[i][1], "molregno": relist2[i][2],
                 "warning_type": relist2[i][3], "warning_class": relist2[i][4], "warning_description": relist2[i][5],
                 "warning_country": relist2[i][6], "warning_year": relist2[i][7], "efo_term": relist2[i][8],
                 "efo_id": relist2[i][9]
                    , "efo_id_for_warning_class": relist2[i][10]})
        i += 1

#     record_dict[str(record_id)] = drug_warning
#     tempdict = copy.deepcopy(record_dict)
#     drug_warning_summary.append(tempdict)
#
# print("warning summary = ", drug_warning_summary[0])
print("===========================================")



'''
遍历 metabolism (新陈代谢)
'''
sql_metabolism = "select drug_record_id from metabolism limit 0,500"
cursor.execute(sql_metabolism)
relist = cursor.fetchall() #获取所有的record_id列表
#接下来去重
relist_no_repeat_metabolism = []
count = 0
while(count <= len(relist)-1):
    if(relist[count] not in relist_no_repeat_metabolism):
        relist_no_repeat_metabolism.append(relist[count])
    count += 1

print("relist value =",relist,"whole length=",len(relist))
print("relist_no_repeat value =",relist_no_repeat_metabolism,"part length=",len(relist_no_repeat_metabolism))

drug_metabolism = {}
# record_dict = {}
# metabolism_summary = []#总表 ，放所有的recordId对应的metabolism_summary

for record in relist_no_repeat_metabolism:
    record_id = record[0]
    sql = "select * from metabolism where drug_record_id = " + "'" + str(record_id) + "'"
    cursor.execute(sql)
    relist2 = cursor.fetchall()  # 同一record_id下可以有零个到多个metabolism
    # print("len(relist2)=",len(relist2))
    i = 0

    # {"record_id":[{drugind_id_1:{drug_indication_value}},{drugind_id_2:{drug_indication_value}}...........]} 每个record_id可能对应零到多个drugind_id

    while (i <= len(relist2) - 1):
        '''
        metabolism = {}
        metabolism["met_id"] = relist2[i][0]
        metabolism["drug_record_id"] = relist2[i][1]
        metabolism["substrate_record_id"] = relist2[i][2]
        metabolism["metabolite_record_id"] = relist2[i][3]
        metabolism["pathway_id"] = relist2[i][4]
        metabolism["pathway_key"] = relist2[i][5]
        metabolism["enzyme_name"] = relist2[i][6]
        metabolism["enzyme_tid"] = relist2[i][7]
        metabolism["met_conversion"] = relist2[i][8]
        metabolism["organism"] = relist2[i][9]
        metabolism["tax_id"] = relist2[i][10]
        metabolism["met_comment"] = relist2[i][11]
        tempdict = copy.deepcopy(metabolism)
        metabolism_list.append(tempdict)
        '''
        if relist2[i][1] not in drug_metabolism:
            drug_metabolism[relist2[i][1]] = [
                {"met_id": relist2[i][0], "drug_record_id": relist2[i][1], "substrate_record_id": relist2[i][2],
                 "metabolite_record_id": relist2[i][3], "pathway_id": relist2[i][4], "pathway_key": relist2[i][5],
                 "enzyme_name": relist2[i][6], "enzyme_tid": relist2[i][7], "met_conversion":relist2[i][8], "organism":relist2[i][9]
                 ,"tax_id":relist2[i][10], "met_comment":relist2[i][11]}]
        else:
            drug_metabolism[relist2[i][1]].append(
                {"met_id": relist2[i][0], "drug_record_id": relist2[i][1], "substrate_record_id": relist2[i][2],
                 "metabolite_record_id": relist2[i][3], "pathway_id": relist2[i][4], "pathway_key": relist2[i][5],
                 "enzyme_name": relist2[i][6], "enzyme_tid": relist2[i][7], "met_conversion": relist2[i][8],
                 "organism": relist2[i][9]
                    , "tax_id": relist2[i][10], "met_comment": relist2[i][11]})


        i += 1

#     record_dict[str(record_id)] = metabolism_list
#     tempdict = copy.deepcopy(record_dict)
#     metabolism_summary.append(tempdict)
#
# print("metabolism_summary = ", metabolism_summary[0])
print("===========================================")

#合并四张record_id_no_repeat表，再二次去重得到recordID总表

full_record_ID = relist_no_repeat_indication + relist_no_repeat_mechanism + relist_no_repeat_warning + relist_no_repeat_metabolism
ultra_record_ID = [] #作为最后的二次去重record_id总表
print("full record ID length=",len(full_record_ID))
j = 0


while(j <= len(full_record_ID)-1):#二次去重
    if (full_record_ID[j] not in ultra_record_ID):
        ultra_record_ID.append(full_record_ID[j])
    j += 1
print("ultra_record_ID length= ",len(ultra_record_ID))


#returnID_fulllist(ID,list) 函数是为了通过record_id在四张字典里面找他们是否存在，存在就返回以ID为keyword的完整列表，不存在就返回0

def returnID_fulllist(ID,drugdict):
    # for ite in list:
    #     if(ite[ID]):#若列表的该字典的keyword=ID存在
    #         return ite
    if(int(ID) in drugdict.keys()):#该类型的drug_dict内存在ID这个KEY,即存在这个record_id对应的值
        return drugdict[int(ID)]
    else:
        return 0
    # count = 0
    # while(count <= len(list)-1):
    #     if(str(ID) in list[count].keys()):##若列表的该字典的keyword=ID存在
    #     # if(ID == list[count][record_id]):
    #         return list[count]
    #     else:
    #         count += 1
    # return 0#若经过上面整个while循环函数都没能返回值的话，那就说明这玩意（ID）就不存在list里面,此时返回0
# debug1 = drug_metabolism.keys()
# debug2 = returnID_fulllist('2468125',drug_metabolism)
# debug3 = drug_metabolism[2468083]

Ultra_Drug = {}#作为最后的 终极四合一总表
for item in ultra_record_ID:
    ID = int(item[0])  # item是一个元组，元组第一个元素就是ID值
    indication_final = returnID_fulllist(ID,drug_indication)
    mechanism_final = returnID_fulllist(ID,drug_mechanism)
    warning_final = returnID_fulllist(ID,drug_warning)
    metabolism_final = returnID_fulllist(ID,drug_metabolism)

    Ultra_Drug["Record_id"] = ID

    if(indication_final != 0):
        Ultra_Drug["drug_indication"] = indication_final
    else:
        Ultra_Drug["drug_indication"] = 'null'

    if (mechanism_final != 0):
        Ultra_Drug["drug_mechanism"] = mechanism_final
    else:
        Ultra_Drug["drug_mechanism"] = 'null'

    if (warning_final != 0):
        Ultra_Drug["drug_warning"] = warning_final
    else:
        Ultra_Drug["drug_warning"] = 'null'

    if (metabolism_final != 0):
        Ultra_Drug["drug_metabolism"] = metabolism_final
    else:
        Ultra_Drug["drug_metabolism"] = 'null'


    devhead.update_one({'Record_id': Ultra_Drug['Record_id']}, {'$set': Ultra_Drug}, True)

# print(Ultra_Drug)


'''




#比如说，找drug_indication_summary 里面的
# IDdict = returnIDdict(1343055,drug_indication_summary)
# print("IDdict = ",IDdict)

# {1343055: [{'drugind_id': 62558, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D001007', 'mesh_heading': 'Anxiety', 'efo_id': 'EFO:0005230', 'efo_term': 'anxiety'}, {'drugind_id': 62559, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D003704', 'mesh_heading': 'Dementia', 'efo_id': 'HP:0000726', 'efo_term': 'Dementia'}, {'drugind_id': 128085, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D003866', 'mesh_heading': 'Depressive Disorder', 'efo_id': 'MONDO:0002050', 'efo_term': 'depressive disorder'}, {'drugind_id': 48687, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('2.0'), 'mesh_id': 'D011565', 'mesh_heading': 'Psoriasis', 'efo_id': 'EFO:0000676', 'efo_term': 'psoriasis'}, {'drugind_id': 43017, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('4.0'), 'mesh_id': 'D011618', 'mesh_heading': 'Psychotic Disorders', 'efo_id': 'EFO:0005407', 'efo_term': 'psychosis'}, {'drugind_id': 139152, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D012559', 'mesh_heading': 'Schizophrenia', 'efo_id': 'MONDO:0005090', 'efo_term': 'schizophrenia'}]}
# {1343055: [{'drugind_id': 62558, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D001007', 'mesh_heading': 'Anxiety', 'efo_id': 'EFO:0005230', 'efo_term': 'anxiety'}, {'drugind_id': 62559, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D003704', 'mesh_heading': 'Dementia', 'efo_id': 'HP:0000726', 'efo_term': 'Dementia'}, {'drugind_id': 128085, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D003866', 'mesh_heading': 'Depressive Disorder', 'efo_id': 'MONDO:0002050', 'efo_term': 'depressive disorder'}, {'drugind_id': 48687, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('2.0'), 'mesh_id': 'D011565', 'mesh_heading': 'Psoriasis', 'efo_id': 'EFO:0000676', 'efo_term': 'psoriasis'}, {'drugind_id': 43017, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('4.0'), 'mesh_id': 'D011618', 'mesh_heading': 'Psychotic Disorders', 'efo_id': 'EFO:0005407', 'efo_term': 'psychosis'}, {'drugind_id': 139152, 'record_id': 1343055, 'molregno': 674902, 'max_phase_for_ind': Decimal('3.0'), 'mesh_id': 'D012559', 'mesh_heading': 'Schizophrenia', 'efo_id': 'MONDO:0005090', 'efo_term': 'schizophrenia'}]}
# 验证成功


Ultra_Drug = {}#作为最后的 终极四合一总表
for item in ultra_record_ID:
    # drug_indication_assemble = []
    # drug_mechanism_assemble = []
    # drug_warning_assemble = []
    # metabolism_assemble = []
    drug_assembl = {}
    ID = item[0]#item是一个元组，元组第一个元素就是ID值
    value_dict1 = returnIDdict(ID,drug_indication_summary)
    value_dict2 = returnIDdict(ID,drug_mechanism_summary)
    value_dict3 = returnIDdict(ID,drug_warning_summary)
    value_dict4 = returnIDdict(ID,metabolism_summary)
    drug_assembl["drug_record_id"] = ID
    if(value_dict1 != 0):#确有其值
        drug_assembl["drug_indication"] = value_dict1
    else:
        drug_assembl["drug_indication"] = 'null'
    if(value_dict2 != 0):
        drug_assembl["drug_mechanism"] = value_dict2
    else:
        drug_assembl["drug_mechanism"] = 'null'
    if (value_dict3 != 0):
        drug_assembl["drug_warning"] = value_dict3
    else:
        drug_assembl["drug_warning"] = 'null'
    if (value_dict4 != 0):
        drug_assembl["metabolism"] = value_dict4
    else:
        drug_assembl["metabolism"] = 'null'
    tempdict = copy.deepcopy(drug_assembl)
    Ultra_Drug.append(tempdict)


#print(Ultra_Drug)

'''





# for element in Ultra_Drug:  # 避免重复存取，只存储没有的，或更新现有的
#     # print('debug_decimal=',element)
#     temp = devhead.update_one({'drug_record_id': element['drug_record_id']}, {'$set': element}, True)











# #sql = "select * from activities limit " + str(start_point) +','+ str(limit)
# sql_indication = "select * from drug_indication limit 0,10"
# cursor.execute(sql_indication)
# relist = cursor.fetchall()
# record_id_indication = []#用来判重，去重record_id
# drug_indication = []#indication 总表
# drug_indication_summary = [] #上述所有drug_indication的总表
# count = 0
# flag = 0
# while(count <= len(relist)-1):
#
#     record_id = relist[count][1]
#     # print("debug=",record_id)
#
#     indication_dict = {}
#     indication_dict["drugind_id"] = relist[count][0]
#     indication_dict["record_id"] = relist[count][1]
#     indication_dict["molregno"] = relist[count][2]
#     indication_dict["max_phase_for_ind"] = relist[count][3]
#     indication_dict["mesh_id"] = relist[count][4]
#     indication_dict["mesh_heading"] = relist[count][5]
#     indication_dict["efo_id"] = relist[count][6]
#     indication_dict["efo_term"] = relist[count][7]
#     tempdict = copy.deepcopy(indication_dict)
#     drug_indication.append(tempdict)
#
#     if(record_id in record_id_indication or len(record_id_indication) == 0):#record_id相同，则放到同一张列表。这里还要考虑第一轮循环列表中还没有ID的情况。
#         # record_id_indication.append(record_id)
#         print("debug=", record_id)
#
#         if(len(record_id_indication) == 0):
#             record_id_indication.append(record_id)
#             flag = 1
#
#     else:
#         '''
#         #两种情况：
#         1. 新的record_id,这时要把上一步老的{"record_id":[{drugind_id_1:{drug_indication_value}},{drugind_id_2:{drug_indication_value}}...........]}做出来
#         2. flag == 1 说明是第一个开始遍历的recordID要特殊处理
#         '''
#         if(flag == 1):#说明上一个recordID是第一个开始遍历的recordID
#             drug_indication_dict = {}
#             drug_indication_dict[record_id_indication[-1]] = drug_indication #record_id_indication[-1] 用来获取列表的最后一个元素
#             tempdict = copy.deepcopy(drug_indication_dict)
#             print("debug!!!!!!!",tempdict)
#             drug_indication_summary.append(tempdict)
#             drug_indication = []
#             flag = 0#这玩意要重新置0
#         else:#说明是新的record_id要把之前旧的存下来
#             record_id_indication.append(record_id)#去重ID表
#             drug_indication_dict = {}
#             drug_indication_dict[record_id] = drug_indication #上一个recordid对应的所有indication
#             tempdict = copy.deepcopy(drug_indication_dict)
#             drug_indication_summary.append(tempdict)
#             drug_indication = []  # 记得要给下一个新的recordID的drug_indication置空
#     count += 1
#
# print("test=",drug_indication_summary[0])
# print("test=",drug_indication_summary[1])
# print("test=",drug_indication_summary[2])
# print("test=",drug_indication_summary[3])
# print("length=",len(drug_indication_summary))




