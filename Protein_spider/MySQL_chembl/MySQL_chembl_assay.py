import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')
mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')

devhead = dtmr_dev.get_collection('Chembl_Assay_test')
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
limit=input("你每轮循环需要几条assay id？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 1401387):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 1401387):
        limit = 1401387 - int(start_point)
    sql = "select * from assays limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    assayinfo_list = cursor.fetchall()
    j = 0
    assay_list = []
    while(j <= len(assayinfo_list)-1):
        assay = {}
        assay_classification = {}
        assay_parameters = {}
        assay["assay_id"] = assayinfo_list[j][0]
        assay["doc_id"] = assayinfo_list[j][1]
        assay["description"] = assayinfo_list[j][2]

        #assay["assay_type"] = assayinfo_list[j][3]#这是一个外键 ，需要去table assay_type进行关联最终生成嵌套字典
        assay_type = {}
        sql_assaytype = "select * from assay_type where assay_type = " + "'" + str(assayinfo_list[j][3]) + "'"
        cursor.execute(sql_assaytype)
        relist = cursor.fetchone()
        assay_type["assay_type"] = relist[0]
        assay_type["assay_desc"] = relist[1]
        tempdict = copy.deepcopy(assay_type)
        assay["assay_type"] = tempdict

        assay["assay_test_type"] = assayinfo_list[j][4]
        assay["assay_category"] = assayinfo_list[j][5]
        assay["assay_organism"] = assayinfo_list[j][6]
        assay["assay_tax_id"] = assayinfo_list[j][7]
        assay["assay_strain"] = assayinfo_list[j][8]
        assay["assay_tissue"] = assayinfo_list[j][9]
        assay["assay_cell_type"] = assayinfo_list[j][10]
        assay["assay_subcellular_fraction"] = assayinfo_list[j][11]

        #assay["tid"] = assayinfo_list[j][12]# 这是一个外键......
        sql_fk = "select * from target_dictionary where tid = " + "'" +str(assayinfo_list[j][12]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        target_info = {}
        target_info["tid"] = relist[0]
        target_info["target_type"] = relist[1] #本来这个也是一个外键（交叉索引里面的交叉索引），但是，如果再进行交叉索引做成嵌套字典那就没啥必要了，毕竟嵌套后在嵌套还不如直接去target总表里面找
        target_info["pref_name"] = relist[2]
        target_info["tax_id"] = relist[3]
        target_info["organism"] = relist[4]
        target_info["chembl_id"] = relist[5]
        target_info["species_group_flag"] = relist[6]
        tempdict = copy.deepcopy(target_info)
        assay["target_info"] = tempdict

        #assay["relationship_type"] = assayinfo_list[j][13]# 这是一个外键......
        sql_fk = "select * from relationship_type where relationship_type = " + "'" + str(assayinfo_list[j][13]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        relationship_type = {}
        relationship_type["relationship_type"] = relist[0]
        relationship_type["relationship_desc"] = relist[1]
        tempdict = copy.deepcopy(relationship_type)
        assay["relationship_type"] = tempdict

        #assay["confidence_score"] = assayinfo_list[j][14]  # 这是一个外键......
        sql_fk = "select * from confidence_score_lookup where confidence_score = " + "'" + str(assayinfo_list[j][14]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        confidence_score = {}
        confidence_score["confidence_score"] = relist[0]
        confidence_score["description"] = relist[1]
        confidence_score["target_mapping"] = relist[2]
        tempdict = copy.deepcopy(confidence_score)
        assay["confidence_score"] = tempdict

        #assay["curated_by"] = assayinfo_list[j][15]  # 这是一个外键......
        sql_fk = "select * from curation_lookup where curated_by = " + "'" + str(assayinfo_list[j][15]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        curated_by = {}
        curated_by["curated_by"] = relist[0]
        curated_by["description"] = relist[1]
        tempdict = copy.deepcopy(curated_by)
        assay["curated_by"] = tempdict



        #assay["src_id"] = assayinfo_list[j][16]  # 这是一个外键......
        sql_fk = "select * from source where src_id = " + "'" + str(assayinfo_list[j][16]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        source = {}
        source["src_id"] = relist[0]
        source["src_description"] = relist[1]
        source["src_short_name"] = relist[2]
        tempdict = copy.deepcopy(source)
        assay["source"] = tempdict

        assay["src_assay_id"] = assayinfo_list[j][17]

        assay["chembl_id"] = assayinfo_list[j][18]  # 这是一个外键......
        sql_fk = "select * from chembl_id_lookup where chembl_id = " + "'" + str(assayinfo_list[j][18]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        chembl_id = {}
        chembl_id["chembl_id"] = relist[0]
        chembl_id["entity_type"] = relist[1]
        chembl_id["entity_id"] = relist[2]
        chembl_id["status"] = relist[3]
        chembl_id["last_active"] = relist[4]
        tempdict = copy.deepcopy(chembl_id)
        assay["chembl_id"] = tempdict

        if(assayinfo_list[j][19]):#cell_id可能为空
            #assay["cell_id"] = assayinfo_list[j][19]  # 这是一个外键......
            sql_fk = "select * from cell_dictionary where cell_id = " + "'" + str(assayinfo_list[j][19]) + "'"
            cursor.execute(sql_fk)
            relist = cursor.fetchone()
            cell_id = {}
            cell_id["cell_id"] = relist[0]
            cell_id["cell_name"] = relist[1]
            cell_id["cell_description"] = relist[2]
            cell_id["cell_source_tissue"] = relist[3]
            cell_id["cell_source_organism"] = relist[4]
            cell_id["cell_source_tax_id"] = relist[5]
            cell_id["clo_id"] = relist[6]
            cell_id["efo_id"] = relist[7]
            cell_id["cellosaurus_id"] = relist[8]
            cell_id["cl_lincs_id"] = relist[9]
            cell_id["chembl_id"] = relist[10]
            cell_id["cell_ontology_id"] = relist[11]
            tempdict = copy.deepcopy(cell_id)
        else:
            tempdict = {}
        assay["cell_id"] = tempdict


        assay["bao_format"] = assayinfo_list[j][20]  # 这是一个外键......
        sql_fk = "select * from bioassay_ontology where bao_id = " + "'" + str(assayinfo_list[j][20]) + "'"
        cursor.execute(sql_fk)
        relist = cursor.fetchone()
        bao_format = {}
        bao_format["bao_id"] = relist[0]
        bao_format["label"] = relist[1]
        tempdict = copy.deepcopy(bao_format)
        assay["bao_format"] = tempdict


        #assay["tissue_id"] = assayinfo_list[j][21]  # 这是一个外键......
        if(assayinfo_list[j][21]):#tissue_id可能为空
            sql_fk = "select * from tissue_dictionary where tissue_id = " + "'" + str(assayinfo_list[j][21]) + "'"
            cursor.execute(sql_fk)
            relist = cursor.fetchone()
            tissue_id = {}
            tissue_id["tissue_id"] = relist[0]
            tissue_id["uberon_id"] = relist[1]
            tissue_id["pref_name"] = relist[2]
            tissue_id["efo_id"] = relist[3]
            tissue_id["chembl_id"] = relist[4]
            tissue_id["bto_id"] = relist[5]
            tissue_id["caloha_id"] = relist[6]
            tempdict = copy.deepcopy(tissue_id)
        else:
            tempdict = {}
        assay["tissue_id"] = tempdict


        #assay["variant_id"] = assayinfo_list[j][22]  # 这是一个外键......
        if(assayinfo_list[j][22]):#variant_id 可能为空
            sql_fk = "select * from variant_sequences where variant_id = " + "'" + str(assayinfo_list[j][22]) + "'"
            cursor.execute(sql_fk)
            relist = cursor.fetchone()
            variant_id = {}
            variant_id["variant_id"] = relist[0]
            variant_id["mutation"] = relist[1]
            variant_id["accession"] = relist[2]
            variant_id["version"] = relist[3]
            variant_id["isoform"] = relist[4]
            variant_id["sequence"] = relist[5]
            variant_id["organism"] = relist[6]
            variant_id["tax_id"] = relist[7]
            tempdict = copy.deepcopy(variant_id)
        else:
            tempdict = {}
        assay["variant_id"] = tempdict

        assay["aidx"] = assayinfo_list[j][23]

        assay_classification_list = [] #单个assay_id可能对应0个到多个assay_class_id
        try:#并不是所有的assay_id都对应有assay_class_id可以从assay_class_map得知
            sql_fk = "select * from assay_class_map where assay_id = " + "'" + str(assayinfo_list[j][0]) + "'"
            cursor.execute(sql_fk)
            relist = cursor.fetchall()#由于在assay_class_map中可知有两个UK:assay_id和assay_class_id,上面的sql只是where 了assay_id,不知道另一个UK:assay_class_id,所以返回的值可能不止一个
            i = 0
            class_id = relist[i][2]
            while(i <= len(relist)-1):
                assay_class_id = class_id#通过这个去assay_classification里面索引
                sql_fk = "select * from assay_classification where assay_class_id = " + "'" + str(assay_class_id) + "'"
                cursor.execute(sql_fk)
                relist = cursor.fetchone()
                assay_classification["assay_class_id"] = assay_class_id
                assay_classification["l1"] = relist[1]
                assay_classification["l2"] = relist[2]
                assay_classification["l3"] = relist[3]
                assay_classification["class_type"] = relist[4]
                assay_classification["source"] = relist[5]
                tempdict = copy.deepcopy(assay_classification)
                assay_classification_list.append(tempdict)
                i += 1
            #print("classification success")
        except:
            assay_classification_list = []
            print("该条参考文献assay数据没有assay_classification字段")
        assay["assay_classification"] = assay_classification_list

        assay_parameters_list = []
        try:#一个assay_id 可能在表assay_parameters中可能对应0到多条assay_param_id
            sql_fk = "select * from assay_parameters where assay_id = " + "'" + str(assayinfo_list[j][0]) + "'"
            cursor.execute(sql_fk)
            relist = cursor.fetchall()
            print("relist=",relist)
            i = 0
            while(i <= len(relist)-1):
                assay_parameters["assay_param_id"] = relist[i][0]
                assay_parameters["assay_id"] = relist[i][1]
                assay_parameters["type"] = relist[i][2]
                assay_parameters["relation"] = relist[i][3]
                assay_parameters["value"] = str(relist[i][4])#decimail数据类型记得转换成字符串
                assay_parameters["units"] = relist[i][5]
                assay_parameters["text_value"] = relist[i][6]
                assay_parameters["standard_type"] = relist[i][7]
                assay_parameters["standard_relation"] = relist[i][8]
                assay_parameters["standard_value"] = str(relist[i][9])
                assay_parameters["standard_units"] = relist[i][10]
                assay_parameters["standard_text_value"] = relist[i][11]
                assay_parameters["comments"] = relist[i][12]
                tempdict = copy.deepcopy(assay_parameters)
                assay_parameters_list.append(tempdict)
                i += 1
            #print("parameters success")
        except:
            assay_parameters_list = []
            print("该条参考文献assay数据没有assay_parameters字段")
        assay["assay_parameters"] = assay_parameters_list
        assay["update_person"] = update_person
        assay["update_time"] = time.asctime()


        tempdict = copy.deepcopy(assay)
        assay_list.append(tempdict)
        j += 1


    print("assay_list info= ", assay_list[0])
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'assay', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}
    print("MySQL_assay_Log =", log)

    start_point = next_start_point  # 下一轮内循环的start point
    for element in assay_list:  # 避免重复存取，只存储没有的，或更新现有的
        #print('debug_decimal=',element)
        temp = devhead.update_one({'assay_id': element['assay_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)





