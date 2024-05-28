import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_Cell_test')
devlog = dtmr_dev.get_collection('Chembl_MySQL_Log')

connection_success = 0
while(connection_success== 0):
    try:
        db = pymysql.connect(host='172.10.10.8', user='root', passwd='Qaz123456', port=3306, database='chembl_32')#charset='utf-8',
        cursor = db.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor
        cursor.execute("SELECT VERSION()")  # 使用 execute()  方法执行 SQL 查询
        data = cursor.fetchone()  # 使用fetchone()方法获取单条(第一条符合execute内语句的)数据
        print('连接成功！',"Database version = %s" % data)
        connection_success = 1
    except:
        print('something wrong! database connection failed')

start_point=input('新的断点从第几条数据开始？？')#初始为0
limit=input("你每轮循环需要几条assay id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 2015):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 2015):
        limit = 2015 - int(start_point)
    sql = "select * from cell_dictionary limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    j = 0
    cell_list = []
    while (j <= len(relist) - 1):
        cell = {}
        cell["cell_id"] = relist[j][0]
        cell["cell_name"] = relist[j][1]
        cell["cell_description"] = relist[j][2]
        cell["cell_source_tissue"] = relist[j][3]
        cell["cell_source_organism"] = relist[j][4]
        cell["cell_source_tax_id"] = relist[j][5]
        cell["clo_id"] = relist[j][6]
        cell["efo_id"] = relist[j][7]
        cell["cellosaurus_id"] = relist[j][8]
        cell["cl_lincs_id"] = relist[j][9]

        cell["chembl_id"] = relist[j][10]
        sql_fk = "select * from chembl_id_lookup where chembl_id = " + "'" + str(relist[j][10]) + "'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchone()
        chembl_id = {}
        chembl_id["chembl_id"] = relist1[0]
        chembl_id["entity_type"] = relist1[1]
        chembl_id["entity_id"] = relist1[2]
        chembl_id["status"] = relist1[3]
        chembl_id["last_active"] = relist1[4]
        tempdict = copy.deepcopy(chembl_id)
        cell["chembl_id"] = tempdict

        cell["cell_ontology_id"] = relist[j][11]
        cell["update_person"] = update_person
        cell["update_time"] = time.asctime()
        tempdict = copy.deepcopy(cell)
        cell_list.append(tempdict)
        j += 1


    print("cell_list info= ", cell_list[0])
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'cell', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}
    print("MySQL_cell_Log =", log)

    start_point = next_start_point  # 下一轮内循环的start point
    for element in cell_list:  # 避免重复存取，只存储没有的，或更新现有的
        #print('debug_decimal=',element)
        temp = devhead.update_one({'cell_id': element['cell_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)










