import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_record_test')
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
limit=input("你每轮循环需要几条record_id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 2590419):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 2590419):
        limit = 2590419 - int(start_point)
    sql = "select * from compound_records limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    record_list = []
    while(j <= len(relist) - 1):
        records = {}
        record_id = relist[j][0]
        records["record_id"] = record_id
        records["molregno"] = relist[j][1] #外键...
        records["doc_id"] = relist[j][2] #外键...
        records["compound_key"] = relist[j][3]
        records["compound_name"] = relist[j][4]

        #records["src_id"] = relist[j][5]#外键 需要索引后做成内嵌字典
        source = {}
        sql_FK = "select * from source where src_id = " + str(relist[j][5])
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchone()
        source["src_id"] = relist[j][5]
        source["src_description"] = relist_FK[1]
        source["src_short_name"] = relist_FK[2]
        tempdict = copy.deepcopy(source)
        records["src_id"] = tempdict

        records["src_compound_id"] = relist[j][6]
        records["cidx"] = relist[j][7]
        records["update_person"] = update_person
        records["update_time"] = time.asctime()
        tempdict = copy.deepcopy(records)
        record_list.append(tempdict)
        j += 1


    print("record_list info= ", record_list[0])
    print("record_list length info= ", len(record_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'record', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time,
           '下次断点设置位置offset = ': next_start_point, '数据库最近一次更新人': update_person}

    print("MySQL_record_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in record_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'record_id': element['record_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)





















