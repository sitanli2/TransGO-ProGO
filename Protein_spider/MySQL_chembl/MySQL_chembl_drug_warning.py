import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_drug_warning_test')
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
limit=input("你每轮循环需要几条drug_warning？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 1636):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 1636):
        limit = 1636 - int(start_point)
    sql = "select * from drug_warning limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    drug_warning_list = []
    while (j <= len(relist) - 1):
        drug_warning = {}
        # i wanna kill myself what the fuck happend last summer? ami in the inferno???
        warning_id = relist[j][0]
        drug_warning["warning_id"] = warning_id
        drug_warning["record_id"] = relist[j][1]
        drug_warning["molregno"] = relist[j][2]
        drug_warning["warning_type"] = relist[j][3]
        drug_warning["warning_class"] = relist[j][4]
        drug_warning["warning_description"] = relist[j][5]
        drug_warning["warning_country"] = relist[j][6]
        drug_warning["warning_year"] = relist[j][7]
        drug_warning["efo_term"] = relist[j][8]
        drug_warning["efo_id"] = relist[j][9]
        drug_warning["efo_id_for_warning_class"] = relist[j][10]

        warning_refs_list = []
        sql_FK = "select * from warning_refs where warning_id = "+"'"+str(warning_id)+"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall()#可能有0个到多个
        count = 0
        while(count <= len(relist1)-1):
            warning_refs = {}
            warnref_id = relist1[count][0]
            warning_refs["warnref_id"] = warnref_id
            warning_refs["warning_id"] = relist1[count][1]
            warning_refs["ref_type"] = relist1[count][2]
            warning_refs["ref_id"] = relist1[count][3]
            warning_refs["ref_url"] = relist1[count][4]
            tempdict = copy.deepcopy(warning_refs)
            warning_refs_list.append(tempdict)
            count += 1

        drug_warning["warning_refs"] = warning_refs_list
        drug_warning["update_person"] = update_person
        drug_warning["update_time"] = time.asctime()

        tempdict = copy.deepcopy(drug_warning)
        drug_warning_list.append(tempdict)
        j += 1

    print("drug_warning_list info= ", drug_warning_list[0])
    print("drug_warning_list length info= ", len(drug_warning_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'drug_warning', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}

    print("MySQL_drug_warning_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in drug_warning_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'warning_id': element['warning_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)