import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_target_test')
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
limit=input("你每轮循环需要几条target？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 15139):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 15139):
        limit = 15139 - int(start_point)
    sql = "select * from target_dictionary limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    target_list = []
    while (j <= len(relist) - 1):
        target = {}
        tid = relist[j][0]
        target["tid"]  = tid

        target["target_type"]  = relist[j][1]#这是外键，索引表为target_type
        sql_fk = "select * from target_type where target_type = "+"'"+str(relist[j][1])+"'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchone()
        target_type = {}
        target_type["target_type"] = relist[j][1]
        target_type["target_desc"] = relist1[1]
        target_type["parent_type"] = relist1[2]
        tempdict = copy.deepcopy(target_type)
        target["target_type"] = tempdict

        target["pref_name"] = relist[j][2]
        target["tax_id"] = relist[j][3]
        target["organism"] = relist[j][4]

        #target["chembl_id"] = relist[j][5] #外键
        sql_fk = "select * from chembl_id_lookup where chembl_id = " + "'" + str(relist[j][5]) + "'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchone()
        chembl_id = {}
        chembl_id["chembl_id"] = relist1[0]
        chembl_id["entity_type"] = relist1[1]
        chembl_id["entity_id"] = relist1[2]
        chembl_id["status"] = relist1[3]
        chembl_id["last_active"] = relist1[4]
        tempdict = copy.deepcopy(chembl_id)
        target["chembl_id"] = tempdict

        target["species_group_flag"] = relist[j][6]

        target_relation_list = []
        sql_FK = "select * from target_relations where tid = " +"'" + str(tid) +"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall() #一个靶点蛋白可以与0个或多个靶点之间储存在联系
        count = 0
        while(count <= len(relist1)-1):
            target_relation = {}
            target_relation["tid"] = relist1[count][0]
            target_relation["relationship"] = relist1[count][1]
            target_relation["related_tid"] = relist1[count][2]
            target_relation["targrel_id"] = relist1[count][3]
            tempdict = copy.deepcopy(target_relation)
            target_relation_list.append(tempdict)
            count += 1

        target["target_relation"] = target_relation_list

        # target_relation_with_component_list = []
        target_relation_with_component = {}
        sql_FK = "select * from target_components where tid = "+"'"+str(tid)+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchone()#有可能存在一对多或多对多的情况？？？

        target_relation_with_component["tid"] = tid
        target_relation_with_component["component_id"] = relist1[1]
        target_relation_with_component["targcomp_id"] = relist1[2]
        target_relation_with_component["homologue"] = relist1[3]
        tempdict = copy.deepcopy(target_relation_with_component)
        target["target_relation_with_component"] = tempdict
        target["update_person"] = update_person
        target["update_time"] = time.asctime()

        # target_relation_with_component_list.append(tempdict)

        # if(len(target_relation_with_component_list) <= 1):
        #     target["target_relation_with_component"] = target_relation_with_component
        # else:
        #     target["target_relation_with_component"] = target_relation_with_component_list


        tempdict = copy.deepcopy(target)
        target_list.append(tempdict)

        j += 1

    print("target_list info= ", target_list[0])
    print("target_list length info= ", len(target_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'target', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}

    print("MySQL_target_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in target_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'tid': element['tid']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)

    # devlog.insert_many()




















