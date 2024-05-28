import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_drug_indication_test')
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
limit=input("你每轮循环需要几条drug_indication？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')


while(int(start_point) < int(count_no) and int(start_point) < 51582):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 51582):
        limit = 51582 - int(start_point)
    sql = "select * from drug_indication limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    drug_indication_list = []
    while (j <= len(relist) - 1):
        drug_indication = {}
        indication_refs = {} #作为drug_indication的索引表，做内嵌字典
        # i wanna kill myself what the fuck happend last summer? ami in the inferno???
        drugind_id = relist[j][0]
        drug_indication["drugind_id"] = drugind_id
        drug_indication["record_id"] = relist[j][1]
        drug_indication["molregno"] = relist[j][2]
        drug_indication["max_phase_for_ind"] = str(relist[j][3])#是decimal类型要转成字符串
        drug_indication["mesh_id"] = relist[j][4]
        drug_indication["mesh_heading"] = relist[j][5]
        drug_indication["efo_id"] = relist[j][6]
        drug_indication["efo_term"] = relist[j][7]

        sql_fk = "select * from indication_refs where drugind_id = "+"'"+str(drugind_id) +"'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchall()#一个drugind_id 可能对应多个indication_refs
        indication_refs = []
        count = 0
        while(count <= len(relist1)-1):
            indication_ref = {}
            indication_ref["indref_id"] = relist1[count][0]
            indication_ref["drugind_id"] = drugind_id
            indication_ref["ref_type"] = relist1[count][2]
            indication_ref["ref_id"] = relist1[count][3]
            indication_ref["ref_url"] = relist1[count][4]
            tempdict = copy.deepcopy(indication_ref)
            indication_refs.append(tempdict)
            count += 1

        drug_indication["indication_refs"] = indication_refs
        drug_indication["update_person"] = update_person
        drug_indication["update_time"] = time.asctime()



        tempdict = copy.deepcopy(drug_indication)
        drug_indication_list.append(tempdict)
        j += 1

    print("drug_indication_list info= ", drug_indication_list[0])
    print("drug_indication_list length info= ", len(drug_indication_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'drug_indication', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}

    print("MySQL_drug_indication_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in drug_indication_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'drugind_id': element['drugind_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)








