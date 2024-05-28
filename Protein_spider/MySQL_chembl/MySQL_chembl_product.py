import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_products_test')
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
limit=input("你每轮循环需要几条products？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 42965):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 42965):
        limit = 42965 - int(start_point)
    sql = "select * from products limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    products_list = []
    while(j <= len(relist) - 1):
        products = {}
        product_id = relist[j][11]
        products["product_id"] = product_id
        products["dosage_form"] = relist[j][0]
        products["route"] = relist[j][1]
        products["trade_name"] = relist[j][2]
        products["approval_date"] = relist[j][3]
        products["ad_type"] = relist[j][4]
        products["oral"] = relist[j][5]
        products["topical"] = relist[j][6]
        products["parenteral"] = relist[j][7]
        products["black_box_warning"] = relist[j][8]
        products["applicant_full_name"] = relist[j][9]
        products["innovator_company"] = relist[j][10]
        products["nda_type"] = relist[j][12]

        product_patents_list = []
        sql_FK = "select * from product_patents where product_id = "+ "'"+str(product_id) +"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall()#一个产品可以有0个或多个专利
        count = 0
        while(count <= len(relist1)-1):
            product_patents = {}
            product_patents["prod_pat_id"] = relist1[count][0]
            product_patents["product_id"] = relist1[count][1]
            product_patents["patent_no"] = relist1[count][2]
            product_patents["patent_expire_date"] = relist1[count][3]
            product_patents["drug_substance_flag"] = relist1[count][4]
            product_patents["drug_product_flag"] = relist1[count][5]

            #product_patents["patent_use_code"] = relist1[count][6]#外键
            sql_fk2 = "select * from patent_use_codes where patent_use_code = "+"'"+str(relist1[count][6])+"'"
            cursor.execute(sql_fk2)
            relist2 = cursor.fetchone()
            patent_use_code = {}
            patent_use_code["patent_use_code"] = relist1[count][6]
            patent_use_code["definition"] = relist2[1]
            tempdict = copy.deepcopy(patent_use_code)
            product_patents["patent_use_code"] = tempdict

            product_patents["delist_flag"] = relist1[count][7]
            product_patents["submission_date"] = relist1[count][8]
            tempdict = copy.deepcopy(product_patents)
            product_patents_list.append(tempdict)
            count += 1
        products["product_patents"] = product_patents_list

        product_formulations_list = []
        sql_FK = "select * from formulations where product_id = "+ "'"+str(product_id) +"'"
        cursor.execute(sql_FK)
        relist1 = cursor.fetchall()#同一种产品可能有0个到多个formulation（配方）
        count = 0
        while(count <= len(relist1)-1):
            product_formulations = {}
            product_formulations["product_id"] = product_id
            product_formulations["ingredient"] = relist1[count][1]
            product_formulations["strength"] = relist1[count][2]
            product_formulations["record_id"] = relist1[count][3]
            product_formulations["molregno"] = relist1[count][4]
            product_formulations["formulation_id"] = relist1[count][5]
            tempdict  = copy.deepcopy(product_formulations)
            product_formulations_list.append(tempdict)
            count += 1
        products["product_formulations"] = product_formulations_list
        products["update_person"] = update_person
        products["update_time"] = time.asctime()



        tempdict = copy.deepcopy(products)
        products_list.append(tempdict)
        j += 1


    print("products_list info= ", products_list[0])
    print("products_list length info= ", len(products_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'products', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time,
           '下次断点设置位置offset = ': next_start_point, '数据库最近一次更新人': update_person}
    print("MySQL_products_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in products_list:  # 避免重复存取，只存储没有的，或更新现有的
        # print('debug_decimal=',element)
        temp = devhead.update_one({'product_id': element['product_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)












