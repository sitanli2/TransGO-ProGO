import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_Component_test')
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
limit=input("你每轮循环需要几条component id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')
while(int(start_point) < int(count_no) and int(start_point) < 10894):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 10894):
        limit = 10894 - int(start_point)
    sql = "select * from component_sequences limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    component_list = []
    while (j <= len(relist) - 1):
        component = {}
        component_synonyms = {}
        component_id = relist[j][0]#这是一个常量 后面还要用就给他直接赋个常量的值
        component["component_id"] = component_id
        component["component_type"] = relist[j][1]
        component["accession"] = relist[j][2]#这个就是uniprot网站的UPID
        component["sequence"] = relist[j][3]
        component["sequence_md5sum"] = relist[j][4]
        component["description"] = relist[j][5]
        component["tax_id"] = relist[j][6]
        component["organism"] = relist[j][7]
        component["db_source"] = relist[j][8]
        component["db_version"] = relist[j][9]

        sql = "select * from component_synonyms where component_id = " + "'" + str(relist[j][0]) + "'"
        cursor.execute(sql)
        relist1 = cursor.fetchall()#一个component_id 可能对应0个到多个component_synonyms即别名。
        i = 0

        synonyms = []
        #print("relist1 = ",relist1)
        while(i <= len(relist1)-1):
            component_synonyms["compsyn_id"] = relist1[i][0]
            component_synonyms["component_id"] = component_id
            component_synonyms["component_synonym"] = relist1[i][2]
            component_synonyms["syn_type"] = relist1[i][3]
            tempdict = copy.deepcopy(component_synonyms)
            synonyms.append(tempdict)
            i += 1
        component["component_synonyms"] = synonyms

        # 下面三个同属需要通过表间桥梁表连接的索引表
        component_relate_go_classification = []  # 之间的关系桥梁表为component_go
        go_classification = {}
        component_relate_domain = []  # 之间的关系桥梁表为component_domain
        domain = {}
        component_relate_protein_classification = []  # 之间的关系桥梁表为component_class
        protein_classification = {}

        sql = "select * from component_go where component_id = "+ "'" + str(component_id) + "'"
        cursor.execute(sql)
        relist1 = cursor.fetchall()#返回的值可能有零个到多个，所以用列表存储
        if(relist1):#判断relist是否为空，为空即上面的sql语句查询不到任何结果
            count = 0
            #print("test=",relist)
            while(count <= len(relist1)-1):
                go_classification = {}
                go_id = relist1[count][2]
                sql_go = "select * from go_classification where go_id = " + "'"+str(go_id)+"'"
                cursor.execute(sql_go)
                relist2 = cursor.fetchone()
                go_classification["go_id"] = go_id
                go_classification["parent_go_id"] = relist2[1]
                go_classification["pref_name"] = relist2[2]
                go_classification["class_level"] = relist2[3]
                go_classification["aspect"] = relist2[4]
                go_classification["path"] = relist2[5]
                tempdict = copy.deepcopy(go_classification)
                component_relate_go_classification.append(tempdict)
                count += 1

        else:
            print("该component与go没有关联")
        component["relation_with_go_classification"] = component_relate_go_classification

        sql = "select * from component_domains where component_id = " + "'" + str(component_id) + "'"
        cursor.execute(sql)
        relist1 = cursor.fetchall()  # 返回的值可能有零个到多个，所以用列表存储
        if (relist1):  # 判断relist是否为空，为空即上面的sql语句查询不到任何结果
            count = 0
            while(count <= len(relist1)-1):
                domain = {}
                domain_id = relist1[count][1]
                sql_domain = "select * from domains where domain_id = "+ "'" + str(domain_id) + "'"
                cursor.execute(sql_domain)
                relist2 = cursor.fetchone()
                domain["domain_id"] = domain_id
                domain["domain_type"] = relist2[1]
                domain["source_domain_id"] = relist2[2]
                domain["domain_name"] = relist2[3]
                domain["domain_description"] = relist2[4]
                tempdict = copy.deepcopy(domain)
                component_relate_domain.append(tempdict)
                count += 1
        else:
            print("该component与domain没有关联")
        component["relation_with_domain"] = component_relate_domain

        sql = "select * from component_class where component_id = " + "'" + str(component_id) + "'"
        cursor.execute(sql)
        relist1 = cursor.fetchall()  # 返回的值可能有零个到多个，所以用列表存储
        if (relist1):  # 判断relist是否为空，为空即上面的sql语句查询不到任何结果
            count = 0
            while (count <= len(relist1) - 1):
                protein_classification = {}
                protein_class_id = relist1[count][1]
                sql_protein = "select * from protein_classification where protein_class_id = " + "'" + str(protein_class_id) + "'"
                cursor.execute(sql_protein)
                relist2 = cursor.fetchone()
                protein_classification["protein_class_id"] = protein_class_id
                protein_classification["parent_id"] = relist2[1]
                protein_classification["pref_name"] = relist2[2]
                protein_classification["short_name"] = relist2[3]
                protein_classification["protein_class_desc"] = relist2[4]
                protein_classification["definition"] = relist2[5]
                protein_classification["class_level"] = relist2[6]
                tempdict = copy.deepcopy(protein_classification)
                component_relate_protein_classification.append(tempdict)
                count += 1
        else:
            print("该component与protein_classification没有关联")
        component["relation_with_protein_classification"] = component_relate_protein_classification

        sql_FK = "select * from biotherapeutic_components where component_id = "+"'"+str(component_id)+"'"
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchall()#以防一对多甚至多对多关系的出现
        count = 0
        biotherapeutics_list = []
        while(count <= len(relist_FK)-1):
            sql_fk = "select * from biotherapeutics where molregno = "+"'"+str(relist_FK[count][1])+"'"
            biotherapeutics = {}
            cursor.execute(sql_fk)
            relist_fk = cursor.fetchone()#唯一主键 molecule_no
            biotherapeutics["molregno"] = relist_fk[0]
            biotherapeutics["description"] = relist_fk[1]
            biotherapeutics["helm_notation"] = relist_fk[2]
            tempdict = copy.deepcopy(biotherapeutics)
            biotherapeutics_list.append(tempdict)
            count += 1
        component["relation_with_biotherapeutics"] = biotherapeutics_list
        component["update_person"] = update_person
        component["update_time"] = time.asctime()


        tempdict = copy.deepcopy(component)
        component_list.append(tempdict)

        j += 1
        #print("j = ",j)

    print("component_list info= ", component_list[0])
    print("component_list_length info= ", len(component_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'component', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time,
           '下次断点设置位置offset = ': next_start_point, '数据库最近一次更新人': update_person}
    print("MySQL_component_Log =", log)


    start_point = next_start_point  # 下一轮内循环的start point

    for element in component_list:  # 避免重复存取，只存储没有的，或更新现有的
        #print('debug_decimal=',element)
        temp = devhead.update_one({'component_id': element['component_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)
















