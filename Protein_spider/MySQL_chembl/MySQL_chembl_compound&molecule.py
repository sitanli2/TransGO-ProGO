import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log

# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')
devhead = dtmr_dev.get_collection('Chembl_compound&molecule_test')
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
limit=input("你每轮循环需要几条molregno？？每多少条数据生成一条日志？？(推荐条数50条)")

count_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')

while(int(start_point) < int(count_no) and int(start_point) < 2268988):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 2268988):
        limit = 2268988 - int(start_point)
    sql = "select * from molecule_dictionary limit " + str(start_point) + ',' + str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall()
    #print("test=",len(relist))
    j = 0
    molecule_list = []
    while(j <= len(relist) - 1):
        molecule_compound = {}
        # i wanna kill myself what the fuck happend last summer? ami in the inferno???
        molregno = relist[j][0]
        molecule_compound["molregno"] = molregno
        molecule_compound["pref_name"] = relist[j][1]

        #molecule_compound["chembl_id"] = relist[j][2]#这是一个外键
        sql_fk = "select * from chembl_id_lookup where chembl_id = " + "'" + str(relist[j][2]) + "'"
        cursor.execute(sql_fk)
        relist1 = cursor.fetchone()
        chembl_id = {}
        chembl_id["chembl_id"] = relist1[0]
        chembl_id["entity_type"] = relist1[1]
        chembl_id["entity_id"] = relist1[2]
        chembl_id["status"] = relist1[3]
        chembl_id["last_active"] = relist1[4]
        tempdict = copy.deepcopy(chembl_id)
        molecule_compound["chembl_id"] = tempdict

        molecule_compound["max_phase"] = str(relist[j][3])
        molecule_compound["therapeutic_flag"] = relist[j][4]
        molecule_compound["dosed_ingredient"] = relist[j][5]
        molecule_compound["structure_type"] = relist[j][6]
        molecule_compound["chebi_par_id"] = relist[j][7]
        molecule_compound["molecule_type"] = relist[j][8]
        molecule_compound["molecule_type"] = relist[j][8]
        molecule_compound["first_approval"] = relist[j][9]
        molecule_compound["oral"] = relist[j][10]
        molecule_compound["parenteral"] = relist[j][11]
        molecule_compound["topical"] = relist[j][12]
        molecule_compound["black_box_warning"] = relist[j][13]
        molecule_compound["first_in_class"] = relist[j][14]
        molecule_compound["chirality"] = relist[j][15]
        molecule_compound["prodrug"] = relist[j][16]
        molecule_compound["inorganic_flag"] = relist[j][17]
        molecule_compound["usan_year"] = relist[j][18]
        molecule_compound["availability_type"] = relist[j][19]
        molecule_compound["usan_stem"] = relist[j][20]
        molecule_compound["polymer_flag"] = relist[j][21]
        molecule_compound["usan_substem"] = relist[j][22]
        molecule_compound["usan_stem_definition"] = relist[j][23]
        molecule_compound["indication_class"] = relist[j][24]
        molecule_compound["withdrawn_flag"] = relist[j][25]

        # 下面几张表同属需要通过表间桥梁表连接的索引表
        molecule_relate_atc_classification = []  # 之间的关系桥梁表为molecule_atc_classification
        atc_classification = {}
        molecule_relate_frac_classification = []  # 之间的关系桥梁表为molecule_frac_classification
        frac_classification = {}
        # molecule_relate_hierarchy = []  # 之间的关系桥梁表为mole_hierarchy 【注】：hierarchy ：等级制度
        molecule_hierarchy = {}

        sql_bridge = "select * from molecule_atc_classification where molregno = " +"'"+str(molregno)+"'"
        cursor.execute(sql_bridge)
        relist2 = cursor.fetchall()
        if(relist2):
            count = 0
            while(count <= len(relist2)-1):
                atc_classification = {}
                level5 = relist2[count][1]
                sql_atc = "select * from atc_classification where level5 = " + "'"+str(level5)+"'"
                cursor.execute(sql_atc)
                relist3 = cursor.fetchone()
                atc_classification["who_name"] = relist3[0]
                atc_classification["level1"] = relist3[1]
                atc_classification["level2"] = relist3[2]
                atc_classification["level3"] = relist3[3]
                atc_classification["level4"] = relist3[4]
                atc_classification["level5"] = relist3[5]
                atc_classification["level1_description"] = relist3[6]
                atc_classification["level2_description"] = relist3[7]
                atc_classification["level3_description"] = relist3[8]
                atc_classification["level4_description"] = relist3[9]
                tempdict = copy.deepcopy(atc_classification)
                molecule_relate_atc_classification.append(tempdict)
                count += 1

        else:
            print("该molecule与atc_clasiffication没有关联")
        molecule_compound["relation_with_atc_classification"] = molecule_relate_atc_classification


        sql_bridge = "select * from molecule_frac_classification where molregno = " + "'" + str(molregno) + "'"
        cursor.execute(sql_bridge)
        relist2 = cursor.fetchall()
        if (relist2):
            count = 0
            while (count <= len(relist2)-1):
                frac_classification = {}
                frac_class_id = relist2[count][1]
                sql_frac = "select * from frac_classification where frac_class_id = " + "'" + str(frac_class_id) + "'"
                cursor.execute(sql_frac)
                relist3 = cursor.fetchone()
                frac_classification["frac_class_id"] = frac_class_id
                frac_classification["active_ingredient"] = relist3[1]
                frac_classification["level1"] = relist3[2]
                frac_classification["level1_description"] = relist3[3]
                frac_classification["level2"] = relist3[4]
                frac_classification["level2_description"] = relist3[5]
                frac_classification["level3"] = relist3[6]
                frac_classification["level3_description"] = relist3[7]
                frac_classification["level4"] = relist3[8]
                frac_classification["level4_description"] = relist3[9]
                frac_classification["level5"] = relist3[10]
                frac_classification["frac_code"] = relist3[11]
                tempdict = copy.deepcopy(frac_classification)
                molecule_relate_frac_classification.append(tempdict)
                count += 1
        else:
            print("该molecule与frac_clasiffication没有关联")
        molecule_compound["relation_with_frac_classification"] = molecule_relate_frac_classification


        sql_hierarchy = "select * from molecule_hierarchy where molregno = " + "'" + str(molregno) + "'"
        #这张表即 molregno -> parent_molregno -> active_molregno 都是molecule_id只不过hierarchy（等级制度不同，有儿子有父亲）
        cursor.execute(sql_hierarchy)
        relist3 = cursor.fetchone()
        molecule_hierarchy["molregno"] = molregno
        molecule_hierarchy["parent_molregno"] = relist3[1]
        molecule_hierarchy["active_molregno"] = relist3[2]
        tempdict = copy.deepcopy(molecule_hierarchy)
        molecule_compound["molecule_hierarchy"] = tempdict

        molecule_relate_hrac_classification = []  # 之间的关系桥梁表为molecule_hrac_classification
        hrac_classification = {}
        sql_bridge = "select * from molecule_hrac_classification where molregno = " + "'" + str(molregno) + "'"
        cursor.execute(sql_bridge)
        relist2 = cursor.fetchall()
        if (relist2):
            count = 0
            while (count <= len(relist2)-1):
                hrac_classification = {}
                hrac_class_id = relist2[count][1]
                sql_hrac = "select * from frac_classification where hrac_class_id = " + "'" + str(hrac_class_id) + "'"
                cursor.execute(sql_hrac)
                relist3 = cursor.fetchone()
                hrac_classification["hrac_class_id"] = hrac_class_id
                hrac_classification["active_ingredient"] = relist3[1]
                hrac_classification["level1"] = relist3[2]
                hrac_classification["level1_description"] = relist3[3]
                hrac_classification["level2"] = relist3[4]
                hrac_classification["level2_description"] = relist3[5]
                hrac_classification["level3"] = relist3[6]
                hrac_classification["hrac_code"] = relist3[7]
                tempdict = copy.deepcopy(hrac_classification)
                molecule_relate_hrac_classification.append(tempdict)
                count += 1
        else:
            print("该molecule与hrac_clasiffication没有关联")
        molecule_compound["relation_with_hrac_classification"] = molecule_relate_hrac_classification



        molecule_relate_irac_classification = []  # 之间的关系桥梁表为molecule_atc_classification
        irac_classification = {}
        sql_bridge = "select * from molecule_irac_classification where molregno = " + "'" + str(molregno) + "'"
        cursor.execute(sql_bridge)
        relist2 = cursor.fetchall()
        if (relist2):
            count = 0
            while (count <= len(relist2)-1):
                irac_classification = {}
                irac_class_id = relist2[count][1]
                sql_irac = "select * from irac_classification where irac_class_id = " + "'" + str(irac_class_id) + "'"
                cursor.execute(sql_irac)
                relist3 = cursor.fetchone()
                irac_classification["irac_class_id"] = irac_class_id
                irac_classification["active_ingredient"] = relist3[1]
                irac_classification["level1"] = relist3[2]
                irac_classification["level1_description"] = relist3[3]
                irac_classification["level2"] = relist3[4]
                irac_classification["level2_description"] = relist3[5]
                irac_classification["level3"] = relist3[6]
                irac_classification["level3_description"] = relist3[7]
                irac_classification["level4"] = relist3[8]
                irac_classification["irac_code"] = relist3[9]
                tempdict = copy.deepcopy(irac_classification)
                molecule_relate_irac_classification.append(tempdict)
                count += 1
        else:
            print("该molecule与irac_clasiffication没有关联")
        molecule_compound["relation_with_irac_classification"] = molecule_relate_irac_classification

        compound_structural_alerts = []#compound_structural_alerts 桥梁表 与structural_alerts相关联
        structural_alerts = {}
        sql_st_alert = "select * from compound_structural_alerts where molregno = " + "'" + str(molregno) + "'"
        cursor.execute(sql_st_alert)
        relist2 = cursor.fetchall()
        if(relist2):
            count = 0
            while(count <= len(relist2)-1):
                structural_alerts = {}
                alert_id = relist2[count][2]
                sql_alert = "select * from structural_alerts where alert_id = " + "'" + str(alert_id) + "'"
                cursor.execute(sql_alert)
                relist3 = cursor.fetchone()
                structural_alerts["alert_id"] = alert_id

                #structural_alerts["alert_set_id"] = relist3[1] #这是一个外键要做成嵌套字典
                alert_sets = {}
                sql_set = "select * from structural_alert_sets where alert_set_id = "+ "'"+str(relist3[1])+"'"
                cursor.execute(sql_set)
                relist4 = cursor.fetchone()
                alert_sets["alert_set_id"] = relist3[1]
                alert_sets["set_name"] = relist4[1]
                alert_sets["set_name"] = relist4[2]
                tempdict = copy.deepcopy(alert_sets)
                structural_alerts["alert_set"] = tempdict

                structural_alerts["alert_name"] = relist3[2]
                structural_alerts["smarts"] = relist3[3]
                tempdict = copy.deepcopy(structural_alerts)
                compound_structural_alerts.append(tempdict)
                count += 1


        else:
            print("该化合物小分子的结构没有alerts字段")
        molecule_compound["relation_with_structural_alerts"] = compound_structural_alerts



        #compound_structures 信息次表 通过molregno 索引后做成内嵌字典!!!!!!!!!!!!!!!!!!!!!!


        '''下面是信息次表的获取'''
        #molecule_synonyms信息次表 通过molregno索引后做成内嵌字典
        molecule_synonyms = [] #一个molregno可能对应0个至多个别名，所以要用列表来存储
        sql_synonyms = "select * from molecule_synonyms where molregno = " +"'" +str(molregno)+ "'"
        cursor.execute(sql_synonyms)
        relist2 = cursor.fetchall()
        if(relist2):
            count = 0
            while(count <= len(relist2)-1):
                synonyms = {}
                synonyms["molregno"] = molregno
                synonyms["syn_type"] = relist2[count][1]
                synonyms["molsyn_id"] = relist2[count][2]
                synonyms["res_stem_id"] = relist2[count][3]
                synonyms["synonyms"] = relist2[count][4]
                tempdict = copy.deepcopy(synonyms)
                molecule_synonyms.append(tempdict)
                count += 1

        else:
            print("该molecule小分子化合物没有别名")
        molecule_compound["molecule_synonyms"] = molecule_synonyms

        #compound_properties 信息次表,通过molregno索引后做成内嵌字典
        #compound_properties = []#属于一对一关系
        properties = {}
        sql_properties = "select * from compound_properties where molregno = " +"'" +str(molregno)+ "'"
        cursor.execute(sql_properties)
        relist2 = cursor.fetchone()
        if(relist2):
            # count = 0
            # while(count <= len(relist2)-1):
                properties = {}
                properties["molregno"] = molregno
                properties["mw_freebase"] = str(relist2[1])#这玩意也是Decimal数据类型，mongo解析不了要转成字符串
                properties["alogp"] = str(relist2[2])
                properties["hba"] = relist2[3]
                properties["hbd"] = relist2[4]
                properties["psa"] = str(relist2[5])
                properties["rtb"] = relist2[6]
                properties["ro3_pass"] = relist2[7]
                properties["num_ro5_violations"] = relist2[8]
                properties["cx_most_apka"] = str(relist2[9])
                properties["cx_most_bpka"] = str(relist2[10])
                properties["cx_logp"] = str(relist2[11])
                properties["cx_logd"] = str(relist2[12])
                properties["molecular_species"] = relist2[13]
                properties["full_mwt"] = str(relist2[14])
                properties["aromatic_rings"] = relist2[15]
                properties["heavy_atoms"] = relist2[16]
                properties["qed_weighted"] = str(relist2[17])
                properties["mw_monoisotopic"] = str(relist2[18])
                properties["full_molformula"] = relist2[19]
                properties["hba_lipinski"] = relist2[20]
                properties["hbd_lipinski"] = relist2[21]
                properties["num_lipinski_ro5_violations"] = relist2[22]
                properties["np_likeness_score"] = str(relist2[23])
                tempdict = copy.deepcopy(properties)
                #compound_properties.append(tempdict)
                # count += 1
        else:
            print("该compound化合物没有properties字段")
        molecule_compound["compound_properties"] = properties


        #compound_structures 信息次表,通过molregno索引后做成内嵌字典
        #compound_structures = []#虽然不可能有多个，但可能有0个的出现且为了避免意外干脆也用列表存算了
        structure = {}
        sql_structure = "select * from compound_structures where molregno = " +"'" +str(molregno)+ "'"
        cursor.execute(sql_structure)
        relist2 = cursor.fetchone()
        if(relist2):
            # count = 0
            # while(count <= len(relist2)-1):
                structure = {}
                structure["molregno"] = molregno
                structure["molfile"] = str(relist2[1])#这个molfile数据是一个很奇怪的东西，数据类型叫做blob类型
                '''
                blob数据类型：
                BLOB (binary large object)----二进制大对象，是一个可以存储二进制文件的容器。
                在计算机中，BLOB常常是数据库中用来存储二进制文件的字段类型。
                BLOB是一个大文件，典型的BLOB是一张图片或一个声音文件，由于它们的尺寸，必须使用特殊的方式来处理（例如：上传、下载或者存放到一个数据库）。
                根据Eric Raymond的说法，处理BLOB的主要思想就是让文件处理器（如数据库管理器）不去理会文件是什么，而是关心如何去处理它。
                但也有专家强调，这种处理大数据对象的方法是把双刃剑，它有可能引发一些问题，如存储的二进制文件过大，会使数据库的性能下降。在数据库中存放体积较大的多媒体对象就是应用程序处理BLOB的典型例子。
                '''

                structure["standard_inchi"] = relist2[2]
                structure["standard_inchi_key"] = relist2[3]
                structure["canonical_smiles"] = relist2[4]
                tempdict = copy.deepcopy(structure)
                # compound_structures.append(tempdict)
                # count += 1

        else:
            print("该compound化合物没有structure字段")
        molecule_compound["compound_structures"] = structure
        molecule_compound["update_person"] = update_person
        molecule_compound["update_time"] = time.asctime()

        tempdict = copy.deepcopy(molecule_compound)
        molecule_list.append(tempdict)
        j += 1

    print("molecule_list info= ", molecule_list[0])
    print("molecule_list length info= ", len(molecule_list))
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'compound&molecule', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time,
           '下次断点设置位置offset = ': next_start_point, '数据库最近一次更新人': update_person}
    print("MySQL_molecule_compound_Log =", log)
    start_point = next_start_point  # 下一轮内循环的start point

    for element in molecule_list:  # 避免重复存取，只存储没有的，或更新现有的
        #print('debug_decimal=',element)
        temp = devhead.update_one({'molregno': element['molregno']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)















