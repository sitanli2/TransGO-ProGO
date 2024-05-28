import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_doc_test')
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
limit=input("你每轮循环需要几条doc_id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')
while(int(start_point) < int(count_no) and int(start_point) < 86361):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 86361):
        limit = 86361 - int(start_point)
    sql = "select * from docs limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    doc_list = []
    while (j <= len(relist) - 1):
        Document = {}
        doc_id = relist[j][0]
        Document["doc_id"] = doc_id
        Document["journal"] = relist[j][1]
        Document["year"] = relist[j][2]
        Document["volume"] = relist[j][3]
        Document["issue"] = relist[j][4]
        Document["first_page"] = relist[j][5]
        Document["last_page"] = relist[j][6]
        Document["pubmed_id"] = relist[j][7]
        Document["doi"] = relist[j][8]

        #Document["chembl_id"] = relist[j][9] #这是一个外键需要关联后存为嵌套字典
        sql_fk = "select * from chembl_id_lookup where chembl_id = " + "'" + str(relist[j][9]) + "'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchone()
        chembl_id = {}
        chembl_id["chembl_id"] = relist1[0]
        chembl_id["entity_type"] = relist1[1]
        chembl_id["entity_id"] = relist1[2]
        chembl_id["status"] = relist1[3]
        chembl_id["last_active"] = relist1[4]
        tempdict = copy.deepcopy(chembl_id)
        Document["chembl_id"] = tempdict

        Document["title"] = relist[j][10]
        Document["doc_type"] = relist[j][11]
        Document["authors"] = relist[j][12]
        Document["abstract"] = relist[j][13]
        Document["patent_id"] = relist[j][14]
        Document["ridx"] = relist[j][15]
        Document["src_id"] = relist[j][16]
        Document["update_person"] = update_person
        Document["update_time"] = time.asctime()

        tempdict = copy.deepcopy(Document)
        doc_list.append(tempdict)
        j += 1
    print("doc_list info= ", doc_list[0])
    print("doc_list length info= ", len(doc_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'doc', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time,
           '下次断点设置位置offset = ': next_start_point, '数据库最近一次更新人': update_person}

    print("MySQL_Doc_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in doc_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'doc_id': element['doc_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)



























