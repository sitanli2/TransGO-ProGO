import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_drug_mechanism_test')
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


start_point=input('新的断点从第几条数据开始？？')#初始为0
limit=input("你每轮循环需要几条drug_mechanism？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 51582):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 51582):
        limit = 51582 - int(start_point)
    sql = "select * from drug_mechanism limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    drug_mechanism_list = []
    while (j <= len(relist) - 1):
        drug_mechanism = {}
        mec_id = relist[j][0]
        drug_mechanism["mec_id"] = mec_id
        drug_mechanism["record_id"] = relist[j][1]
        drug_mechanism["molregno"] = relist[j][2]
        drug_mechanism["mechanism_of_action"] = relist[j][3]
        drug_mechanism["tid"] = relist[j][4]

        #drug_mechanism["site_id"] = relist[j][5]#外键 关联binding_sites表，将其做成内嵌字典
        sql_FK = "select * from binding_sites where site_id = "+"'"+str(relist[j][5])+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchone()#relist[j][5]可能有NULL的出现
        if(relist1):#site_id不为null
            site = {}
            site["site_id"] = relist1[0]
            site["site_name"] = relist1[1]
            site["tid"] = relist1[2]
            tempdict = copy.deepcopy(site)
        else:
            print("该Drug_mechanism 的site 为空")
            tempdict = {}
        drug_mechanism["site_id"] = tempdict

        #drug_mechanism["action_type"] = relist[j][6] #外键 关联action_type表，将其做成内嵌字典
        sql_FK = "select * from action_type where action_type = "+"'"+str(relist[j][6])+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchone()
        if(relist1):#以防万一，这个也没有。判断一下
            action = {}
            action["action_type"] = relist[j][6]
            action["description"] = relist1[1]
            action["description"] = relist1[2]
            tempdict = copy.deepcopy(action)
        else:
            tempdict = {}
        drug_mechanism["action_type"] = tempdict

        drug_mechanism["direct_interaction"] = relist[j][7]
        drug_mechanism["molecular_mechanism"] = relist[j][8]
        drug_mechanism["disease_efficacy"] = relist[j][9]
        drug_mechanism["mechanism_comment"] = relist[j][10]
        drug_mechanism["selectivity_comment"] = relist[j][11]
        drug_mechanism["binding_site_comment"] = relist[j][12]

        #drug_mechanism["variant_id"] = relist[j][13] #外键 关联variant_sequence,将其作为内嵌字典
        sql_FK = "select * from variant_sequences where variant_id = "+"'"+str(relist[j][13])+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchone()
        if (relist1):  # 以防万一，这个也没有。判断一下
            variant_sequence = {}
            variant_sequence["action_type"] = relist[j][13]
            variant_sequence["mutation"] = relist1[1]
            variant_sequence["accession"] = relist1[2]
            variant_sequence["version"] = relist1[3]
            variant_sequence["isoform"] = relist1[4]
            variant_sequence["sequence"] = str(relist1[5]) #这个内容是一个blob数据类型，要把它转字符串
            variant_sequence["organism"] = relist1[6]
            variant_sequence["tax_id"] = relist1[7]
            tempdict = copy.deepcopy(variant_sequence)
        else:
            tempdict = {}
        drug_mechanism["variant_id"] = tempdict

        sql_FK = "select * from mechanism_refs where mec_id = "+"'"+str(mec_id) +"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall()  # mec_id 可能对应多个mechanism_refs
        mechanism_refs = []
        count = 0
        while (count <= len(relist1) - 1):
            mechanism_ref = {}
            mechanism_ref["mecref_id"] = relist1[count][0]
            mechanism_ref["mec_id"] = mec_id
            mechanism_ref["ref_type"] = relist1[count][2]
            mechanism_ref["ref_id"] = relist1[count][3]
            mechanism_ref["ref_url"] = relist1[count][4]
            tempdict = copy.deepcopy(mechanism_ref)
            mechanism_refs.append(tempdict)
            count += 1
        drug_mechanism["mechanism_refs"] = mechanism_refs
        drug_mechanism["update_person"] = update_person
        drug_mechanism["update_time"] = time.asctime()


        tempdict = copy.deepcopy(drug_mechanism)
        drug_mechanism_list.append(tempdict)
        j += 1
    print("drug_mechanism_list info= ", drug_mechanism_list[0])
    print("drug_mechanism_list length info= ", len(drug_mechanism_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'drug_mechanism', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}

    print("MySQL_drug_mechanism_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in drug_mechanism_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'mec_id': element['mec_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)














