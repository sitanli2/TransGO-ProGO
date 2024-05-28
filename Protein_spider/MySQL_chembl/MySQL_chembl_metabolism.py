import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_metabolism_test')
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
limit=input("你每轮循环需要几条met_id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 2126):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 2126):
        limit = 2126 - int(start_point)
    sql = "select * from metabolism limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    metabolism_list = []
    while(j <= len(relist)-1 ):
        metabolism = {}

        met_id = relist[j][0]
        metabolism["met_id"] = met_id
        metabolism["drug_record_id"] = relist[j][1]
        metabolism["substrate_record_id"] = relist[j][2]
        metabolism["metabolite_record_id"] = relist[j][3]
        metabolism["pathway_id"] = relist[j][4]
        metabolism["pathway_key"] = relist[j][5]
        metabolism["enzyme_name"] = relist[j][6]
        metabolism["enzyme_tid"] = relist[j][7] #这是一个外键，关联索引target表，由于是几大主表之一，不用再内嵌字典。
        metabolism["met_conversion"] = relist[j][8]
        metabolism["organism"] = relist[j][9]
        metabolism["tax_id"] = relist[j][10]
        metabolism["met_comment"] = relist[j][11]

        metabolism_refs_list = []
        sql_FK = "select * from metabolism_refs where met_id = "+"'" +str(met_id)+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall()#同一个met_id可能对应了多个metabolism_refs
        count = 0
        while(count <= len(relist1)-1):
            metabolism_refs = {}
            metref_id = relist1[count][0]
            metabolism_refs["metref_id"] = metref_id
            metabolism_refs["met_id"] = met_id
            metabolism_refs["ref_type"] = relist1[count][2]
            metabolism_refs["ref_id"] = relist1[count][3]
            metabolism_refs["ref_url"] = relist1[count][4]
            tempdict = copy.deepcopy(metabolism_refs)
            metabolism_refs_list.append(tempdict)
            count += 1
        metabolism["metabolism_refs"] = metabolism_refs_list
        metabolism["update_person"] = update_person
        metabolism["update_time"] = time.asctime()
        tempdict = copy.deepcopy(metabolism)
        metabolism_list.append(tempdict)
        j += 1


    print("metabolism_list info= ", metabolism_list[0])
    print("metabolism_list length info= ", len(metabolism_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'metabolism', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}

    print("MySQL_metabolism_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in metabolism_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'met_id': element['met_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)

















