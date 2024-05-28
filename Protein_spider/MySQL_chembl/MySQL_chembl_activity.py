#chembl官方数据库解释文档 ：https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.txt

import pymysql
from pymongo import MongoClient
import copy
import time #记录时间存入log


# mongoClient = MongoClient('mongodb://localhost:27017/')
# dtmr_dev = mongoClient.get_database('dtmr_dev')

mongoClient = MongoClient('mongodb://172.10.10.9:27017/')
dtmr_dev = mongoClient.get_database('dtmr_dev')
dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')

devhead = dtmr_dev.get_collection('Chembl_Activity_test')
devlog = dtmr_dev.get_collection('Chembl_MySQL_Log')
devstd_lookup = dtmr_dev.get_collection('Chembl_Activity_stds_lookup')


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
limit=input("你每轮循环需要几条activities id？？每多少条数据生成一条日志？？(推荐条数50条)")

activity_no = input('你此次运行想要到第几条为止？？')
update_person = input('你的姓名是？？')


'''先存解释表：ACTIVITY_STDS_LOOKUP （一般解释表表项都比较少，所以可以一次性存完，不需要断点）
注意在这张表的官方解释文档里 有两个UK即unique key，这说明想确定唯一表项需要STANDARD_TYPE & STANDARD_UNITS两个UK
Table storing details of standardised activity types and units, with permitted value ranges. 
Used to determine the correct standard_type and standard_units for a range of published types/units.
'''
sql_stds_lookup = "select * from activity_stds_lookup"
cursor.execute(sql_stds_lookup)
relist0 = cursor.fetchall()
stds_list = []
j = 0
while(j <= len(relist0)-1):
    stds_dict = {}
    stds_dict["std_act_id"] = relist0[j][0]
    stds_dict["standard_type"] = relist0[j][1]
    stds_dict["definition"] = relist0[j][2]
    stds_dict["standard_units"] = relist0[j][3]
    stds_dict["normal_range_min"] = str(relist0[j][4])
    stds_dict["normal_range_max"] = str(relist0[j][5])
    j += 1
    tempdict = copy.deepcopy(stds_dict)
    stds_list.append(tempdict)
#devstd_lookup.insert_many(stds_list)
for element in stds_list:  # 避免重复存取，只存储没有的，或更新现有的
    temp = devstd_lookup.update_one({'std_act_id': element['std_act_id']}, {'$set': element}, True)
print("解释表sql_stds_lookup 已经就绪")



while(int(start_point) < int(activity_no) and int(start_point) < 19995883):#19995883 为chembl数据库内activate 总条数。
    if(int(start_point)+int(limit) >= 19995883):
        limit = 19995883 - int(start_point)

    '''
    select * from table_name limit 0,10
    0代表从第0条记录后面开始，也就是从第一条开始。
    因此在MySQL中的断点方式找到了
    '''

    sql = "select * from activities limit " + str(start_point) +','+ str(limit)
    cursor.execute(sql)
    relist = cursor.fetchall() #接收所有activity_supp_map返回的行
    '''
    fetchone()
    功能：获取下一个查询结果集，结果集是一个对象
    fetchall()
    功能：接收全部的返回的行
    '''
    activities_id_list = []
    j = 0
    activitise_list = []
    while(j <= len(relist)-1):
        activity = {}
        # activiy_properties = {}
        # activity_supp = {}
        activity["activity_id"] = relist[j][0]  # 靠activity_id去到activity表中关联大字典activity，去activity_properties表中关联内嵌字典activiy_properties
        activity["assay_id"] = relist[j][1]  # 靠assay_id去到assay表中关联到target_id，再把target_id加到主表activity中去
        activity["doc_id"] = relist[j][2]
        activity["record_id"] = relist[j][3]
        activity["molregno"] = relist[j][4]

        # 这里还要加一条target id,得通过上面的assayid 去到assays表里面有个tid就是targetid
        sql1 = "select tid from assays where assay_id = " + str(relist[j][1])
        cursor.execute(sql1)
        relist1 = cursor.fetchone()
        activity["target_id"] = relist1[0]

        activity["standard_relation"] = relist[j][5]
        activity["standard_value"] = str(relist[j][6])# Decimal是一种数字类型float是单精度，double是双精度，decimal是数字型，它们所占的内存空间不一样，表示的位数也不一样。但是在mongo里面无法将其解码为json所以要把它转成字符型
        activity["standard_units"] = relist[j][7]
        activity["standard_flag"] = relist[j][8]
        activity["standard_type"] = relist[j][9]
        activity["activity_comment"] = relist[j][10]

        #activity["data_validity_comment"] = relist[j][11]#这个是一个外键，需要做成字典。
        data_validity_comment = {}
        if(relist[j][11]):
            sql_FK = "select * from data_validity_lookup where data_validity_comment = " + "'" + str(relist[j][11]) + "'"
            cursor.execute(sql_FK)
            relist_FK = cursor.fetchone()  # 唯一标识键为data_validity_comment，所以只会返回一条
            data_validity_comment["data_validity_comment_value"] = relist[j][11]
            data_validity_comment["description"] = relist_FK[1]
            tempdict = copy.deepcopy(data_validity_comment)
        else:#为空则插入空字典
            tempdict = {}
        activity["data_validity_comment"] = tempdict


        activity["potential_duplicate"] = relist[j][12]
        activity["pchembl_value"] = str(relist[j][13])

        #activity["bao_endpoint"] = relist[j][14]#bao_endpoint 也是一个外键，需要关联后做成内嵌字典
        bao_endpoint = {}
        sql_FK = "select * from bioassay_ontology where bao_id = " + "'" +relist[j][14] + "'"
        #sql语句 bao_id = 后面的字符串必须要加引号，否则就会报错
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchone()
        bao_endpoint["bao_endpoint_id"] = relist_FK[0]
        bao_endpoint["label"] = relist_FK[1]
        tempdict = copy.deepcopy(bao_endpoint)
        activity["bao_endpoint"] = tempdict

        activity["uo_units"] = relist[j][15]
        activity["qudt_units"] = relist[j][16]
        activity["toid"] = relist[j][17]
        activity["upper_value"] = relist[j][18]
        activity["standard_upper_value"] = relist[j][19]

        #activity["src_id"] = relist[j][20]#src_id 也是一个外键，需要关联后做成内嵌字典
        source = {}
        sql_FK = "select * from source where src_id = " + str(relist[j][20])
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchone()
        source["src_id"] = relist[j][20]
        source["src_description"] = relist_FK[1]
        source["src_short_name"] = relist_FK[2]
        tempdict = copy.deepcopy(source)
        activity["Source_info"] = tempdict

        activity["type"] = relist[j][21]
        activity["relation"] = relist[j][22]
        activity["value"] = str(relist[j][23])
        activity["units"] = relist[j][24]
        activity["text_value"] = relist[j][25]
        activity["standard_text_value"] = relist[j][26]

        activiy_propertie_list = []
        # activiy_properties = {}
        sql_fk = "select * from activity_properties where activity_id = " + str(relist[j][0])
        cursor.execute(sql_fk)
        relist_fk = cursor.fetchall()#一个activity 可能对应零个至多个properties
        count = 0
        while(count <= len(relist_fk)-1):
            activiy_properties = {}
            # cursor.execute(sql2)
            # relist1 = cursor.fetchall()#一个activity 可能对应零个至多个properties
            # # print("select * from activity_properties where activity_id =",relist3)
            activiy_properties["ap_id"] = relist_fk[count][0]
            # activiy_properties["activity_id"] = relist3[1] 其实activity_id就是这个嵌套字典的父字典的id不用再存一次
            activiy_properties["Type"] = relist_fk[count][2]
            activiy_properties["Relation"] = relist_fk[count][3]
            activiy_properties["Value"] = str(relist_fk[count][4])
            activiy_properties["Units"] = relist_fk[count][5]
            activiy_properties["text_value"] = relist_fk[count][6]
            activiy_properties["Standard_type"] = relist_fk[count][7]
            activiy_properties["Standard_relation"] = relist_fk[count][8]
            activiy_properties["Standard_value"] = str(relist_fk[count][9])
            activiy_properties["Standard_units"] = relist_fk[count][10]
            activiy_properties["Standard_text_value"] = relist_fk[count][11]
            activiy_properties["comments"] = relist_fk[count][12]
            activiy_properties["result_flag"] = relist_fk[count][13]
            tempdict = copy.deepcopy(activiy_properties)
            activiy_propertie_list.append(tempdict)
            count += 1
        # except:
        #     print("该条活性物activity数据没有properties字段")
        #     activiy_properties = {}
        activity["activity_properties"] = activiy_propertie_list




        activiy_supp_list = []
        # activity_supp = {}
        sql_FK = "select * from activity_supp_map where activity_id = " + str(relist[j][0])
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchall() #接收activity_supp_map返回的行即smid,同一个activity可能对应0个到多个smid
        if(relist_FK): #可能为空，以防万一
            count = 0
            while(count <= len(relist_FK)):
                sql_fk = "select * from activity_supp where smid = "+"'"+str(relist_FK[count][2])+"'"
                cursor.execute(sql_fk)
                relist_fk = cursor.fetchone()
                activity_supp = {}
                activity_supp["as_id"] = relist_fk[0]
                activity_supp["rgid"] = relist_fk[1]
                activity_supp["smid"] = relist_fk[2]
                activity_supp["type"] = relist_fk[3]
                activity_supp["relation"] = relist_fk[4]
                activity_supp["value"] = relist_fk[5]
                activity_supp["units"] = relist_fk[6]
                activity_supp["text_value"] = relist_fk[7]
                activity_supp["standard_type"] = relist_fk[8]
                activity_supp["standard_relation"] = relist_fk[9]
                activity_supp["standard_value"] = relist_fk[10]
                activity_supp["standard_units"] = relist_fk[11]
                activity_supp["standard_text_value"] = relist_fk[12]
                activity_supp["comments"] = relist_fk[13]
                tempdict = copy.deepcopy(activity_supp)
                activiy_supp_list.append(tempdict)
                count += 1
        activity["activiy_supp"] = activiy_supp_list

        sql_FK = "select * from ligand_eff where activity_id = "+"'"+str(relist[j][0])+"'"
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchone()#一一对应
        ligand_eff = {}

        if(relist_FK):
            ligand_eff["activity_id"] = relist[j][0]
            ligand_eff["bei"] = str(relist_FK[1])
            ligand_eff["sei"] = str(relist_FK[2])
            ligand_eff["le"] = str(relist_FK[3])
            ligand_eff["lle"] = str(relist_FK[4])
            tempdict = copy.deepcopy(ligand_eff)
        else:
            tempdict = {}

        activity["ligand_eff"] = tempdict

        predicted_binding_domains_list = []
        sql_FK = "select * from predicted_binding_domains where activity_id = "+"'"+str(relist[j][0])+"'"
        cursor.execute(sql_FK)
        relist_FK = cursor.fetchall()#activity与binding_site相关联的关系可以是多对多
        count = 0
        while(count <= len(relist_FK)-1):
            predicted_binding_domains = {}
            predicted_binding_domains["predbind_id"] = relist_FK[count][0]
            predicted_binding_domains["activity_id"] = relist_FK[count][1]

            predicted_binding_domains["site_id"] = relist_FK[count][2] #这是一个外键 关联表binding_sites
            sql_fk = "select * from binding_sites where site_id = "+"'"+str(relist_FK[count][2])+"'"
            cursor.execute(sql_fk)
            relist_fk = cursor.fetchone()
            binding_site = {}
            binding_site["site_id"] = sql_fk[0]
            binding_site["site_name"] = sql_fk[1]
            binding_site["tid"] = sql_fk[2]
            tempdict = copy.deepcopy(binding_site)

            predicted_binding_domains["binding_site"] = tempdict
            predicted_binding_domains["prediction_method"] = relist_FK[count][3]
            predicted_binding_domains["confidence"] = relist_FK[count][4]
            tempdict = copy.deepcopy(predicted_binding_domains)
            predicted_binding_domains_list.append(tempdict)
            count += 1
        activity["predicted_binding_domains"] = predicted_binding_domains_list
        activity["update_person"] = update_person
        activity["update_time"] = time.asctime()



        #print('relist2=',relist2)
        # if(relist_FK):#因为不是所有的activityid都有对应的smid，relist2可能为空
        #     sql4 = "select * from activity_supp where smid = " + str(relist2[0])  # 通过smid去到activity_supp 关联supp信息
        #
        # #sql4 = "select * from activity_supp where smid = 2938"
        # try:
        #     cursor.execute(sql4)
        #     relist3 = cursor.fetchone()# 接收activity_supp返回的信息即activity_supp
        #     # print("test=",relist4)
        #     activity_supp["as_id"] = relist3[0]
        #     activity_supp["rg_id"] = relist3[1]
        #     activity_supp["sm_id"] = relist3[2]
        #     activity_supp["type"] = relist3[3]
        #     activity_supp["relation"] = relist3[4]
        #     activity_supp["value"] = relist3[5]
        #     activity_supp["units"] = relist3[6]
        #     activity_supp["text_value"] = relist3[7]
        #     activity_supp["standard_type"] = relist3[8]
        #     activity_supp["standard_relation"] = relist3[9]
        #     activity_supp["standard_value"] = relist3[10]
        #     activity_supp["standard_units"] = relist3[11]
        #     activity_supp["standard_text_value"] = relist3[12]
        #     activity_supp["comments"] = relist3[13]
        # except:
        #     print("该条活性物activity数据没有activity_supp字段")
        #     activity_supp = {}
        # activity["activity_supp"] = activity_supp

        activities_id_list.append(relist[j][0])
        j += 1
        tempdict = copy.deepcopy(activity)
        activitise_list.append(tempdict)

    print("activities_id_list info =",activities_id_list)
    print("activities_list info= ",activitise_list[0])
    next_start_point = int(start_point) + int(limit)
    excute_time = time.asctime()
    log = {'Log_type': 'activity', '最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point,'数据库最近一次更新人':update_person}
    print("MySQL_Log =",log)

    start_point = next_start_point  # 下一轮内循环的start point
    #count = 0
    for element in activitise_list:  # 避免重复存取，只存储没有的，或更新现有的
        #count+=1
        #print("activity_list的第%d个",count)
        #print("activity_list的错误：",element)
        temp = devhead.update_one({'activity_id': element['activity_id']}, {'$set': element}, True)
    devlog.update_one({'Log_type':log['Log_type']},{'$set':log},True)





'''

sql = "select * from activity_supp_map limit " + str(start_point) +','+ str(limit)

cursor.execute(sql)
relist = cursor.fetchall() #接收所有activity_supp_map返回的行

activities_id_list = []
i = 0
activitise_list = []
while(i <= len(relist)-1):

    sql1 = "select * from activities where activity_id = "+ str(relist[i][1])
    cursor.execute(sql1)
    relist1 = cursor.fetchall()
    j = 0
    while(j <= len(relist1)-1):
        activity = {}
        activiy_properties = {}
        activity_supp = {}
        activity["activity_id"] = relist1[j][0]#靠activity_id去到activity表中关联大字典activity，去activity_properties表中关联内嵌字典activiy_properties
        activity["assay_id"] = relist1[j][1]#靠assay_id去到assay表中关联到target_id
        activity["doc_id"] = relist1[j][2]
        activity["record_id"] = relist1[j][3]
        activity["molregno"] = relist1[j][4]

        #这里还要加一条target id,得通过上面的assayid 去到assays表里面有个tid就是targetid
        sql2 = "select tid from assays where assay_id = " + str(relist1[j][1])
        cursor.execute(sql2)
        relist2 = cursor.fetchone()
        activity["target_id"] = relist2[0]

        activity["standard_relation"] = relist1[j][5]
        activity["standard_value"] = relist1[j][6]
        activity["standard_units"] = relist1[j][7]
        activity["standard_flag"] = relist1[j][8]
        activity["standard_type"] = relist1[j][9]
        activity["activity_comment"] = relist1[j][10]
        activity["data_validity_comment"] = relist1[j][11]
        activity["potential_duplicate"] = relist1[j][12]
        activity["pchembl_value"] = relist1[j][13]
        activity["bao_endpoint"] = relist1[j][14]
        activity["uo_units"] = relist1[j][15]
        activity["qudt_units"] = relist1[j][16]
        activity["toid"] = relist1[j][17]
        activity["upper_value"] = relist1[j][18]
        activity["standard_upper_value"] = relist1[j][19]
        activity["src_id"] = relist1[j][20]
        activity["type"] = relist1[j][21]
        activity["relation"] = relist1[j][22]
        activity["value"] = relist1[j][23]
        activity["units"] = relist1[j][24]
        activity["text_value"] = relist1[j][25]
        activity["standard_text_value"] = relist1[j][26]

        sql3 = "select * from activity_properties where activity_id = "+ str(relist1[j][0])

        try:
            cursor.execute(sql3)
            relist3 = cursor.fetchone()
            #print("select * from activity_properties where activity_id =",relist3)
            activiy_properties["ap_id"] = relist3[0]
            # activiy_properties["activity_id"] = relist3[1] 其实activity_id就是这个嵌套字典的父字典的id不用再存一次
            activiy_properties["Type"] = relist3[2]
            activiy_properties["Relation"] = relist3[2]
            activiy_properties["Value"] = relist3[3]
            activiy_properties["Units"] = relist3[4]
            activiy_properties["text_value"] = relist3[5]
            activiy_properties["Standard_type"] = relist3[6]
            activiy_properties["Standard_relation"] = relist3[7]
            activiy_properties["Standard_value"] = relist3[8]
            activiy_properties["Standard_units"] = relist3[9]
            activiy_properties["Standard_text_value"] = relist3[10]
            activiy_properties["comments"] = relist3[11]
            activiy_properties["result_flag"] = relist3[12]
        except:
            print("该条活性物activity数据没有properties字段")
            activiy_properties = {}
        activity["activity_properties"] = activiy_properties

        print("relist=", relist)
        sql4 = "select * from activity_supp where smid = " + str(relist[i][2])#通过smid去到activity_supp 关联supp信息
        try:
            cursor.execute(sql4)
            relist4 = cursor.fetchone()
            #print("test=",relist4)
            activity_supp["as_id"] = relist4[0]
            activity_supp["rg_id"] = relist4[1]
            activity_supp["sm_id"] = relist4[2]
            activity_supp["type"] = relist4[3]
            activity_supp["relation"] = relist4[4]
            activity_supp["value"] = relist4[5]
            activity_supp["units"] = relist4[6]
            activity_supp["text_value"] = relist4[7]
            activity_supp["standard_type"] = relist4[8]
            activity_supp["standard_relation"] = relist4[9]
            activity_supp["standard_value"] = relist4[10]
            activity_supp["standard_units"] = relist4[11]
            activity_supp["standard_text_value"] = relist4[12]
            activity_supp["comments"] = relist4[13]
        except:
            print("该条活性物activity数据没有activity_supp字段")
            activity_supp = {}
        activity["activity_supp"] = activity_supp



        activities_id_list.append(relist1[j][0])

        j += 1
        tempdict = copy.deepcopy(activity)
        activitise_list.append(tempdict)
    i += 1


print("activities_id_list info =",activities_id_list)
print("activities_list info= ",activitise_list[0])

next_start_point = int(start_point) + int(limit)
excute_time = time.asctime()
log = {'最新设置断点位置': start_point, '最新爬取条数': limit, '该日志创建时间': excute_time, '下次断点设置位置offset = ': next_start_point}
print("MySQL_Log =",log)

'''


'''
为了方便之后在mongo中的查询 chembl 官方数据库 的table表我将它分为以下几类放入mongoDB
1. 最重要的信息总表：
如activities，assays.....drugs
作为存入mongoDB的几大主表和次表，主键id就为对应的activity_id.....

2. 信息次表：
如activity_supp,activity_properties.....
他们则需作为主表activity的内嵌字典，至于主表activity如何关联到activity_supp,activity_properties.....等次表则是看各种外键的关系
如若想得到 activity_id = 31873 对应的activity_properties则只需直接 select * from activity_properties where activity_id = 31873
有些没这么好茶可能要查两次
如若想得到 activity_id = 31873 对应的activity_supp，要先select smid from activity_supp_map where activity_id = 31873 得到smid
再去到activity_supp里面去 select * from activity_supp where smid = ....上一步得到的smid 
最后将这个smid对应的activity_supp 内容作为内嵌字典存入activity中去

3. “解释”表（一般表名后面都会带一个lookup后缀 ，如stds_lookup）
由于整个数据库中存在大量重复且长串的数据或集合，为了防止浪费存储资源，chembl数据库采用了如下方法：
只存储某个数据集对应的编号或别名
如activity 表中的standard_type = IC50 ,这个IC50 就是这个type的别名，更据IC50 以及standard_units=nM 去activity_stds_lookup里面关联
select * from activity_stds_lookup where standard_type = IC50,standard_units=nM 得到整条standard完整的解释
所以stds_lookup其实就是standard的解释表关联外键就为standard_type ,standard_units
所以stds_lookup 要作为一个与主表activity 并列的字典存入mongo中，方便后续的mongoDB里面查询IC50....等standard具体是什么意思。

4.桥梁/索引表，以及各表中存在的外键：

4.1 桥梁表/索引表/表间桥梁
比如在assay_class_map中可知有两个UK:assay_id和assay_class_id,sql若只是where 了assay_id,不知道另一个UK:assay_class_id,返回的值则可能不止一个，
这个assay_class_map其实就是索引表，可以把它看作一道桥梁，通过assay_id和assay_class_id 连接assays和assay_classification这两张表

4.2 各表中存在的外键
如 activity表中存在一个外键叫src_id ->,bao_endpoint->......
以src_id="1" 为例 此时需要把src_id 下面的内容作为内嵌字典存入activity下
总而言之，只要是外键，即与其他表有关联关系的键，则需要通过这个外键去到相应的外表拿到它的完整数据并作为嵌套字典存储到原外键的位置（因为不同于MySQL，mongoDB并不是关系型数据库需要我来给他们搭建关系）
所以activity作为这个数据库的主表，因为他有其他大部分次表的外键（除了drug_id,cell_id等等,因此drug表需要独立与activity表及其关系次表之外另寻关系。比如drug显然和小分子化合物molecule有关）

#外键索引内容中的外键：本来这个也是一个外键（交叉索引里面的交叉索引），但是，如果再进行交叉索引做成嵌套字典的嵌套字典那就没啥必要了，毕竟嵌套后再嵌套还不如直接去mongo对应的总表里面找来的方便且美观
【注】：当外键为activity、target、assay、molecule&compound、DOc...这几张主表时，可以不需要再去索引，因为在MongoDB里面我们已经存了他们的总表，只需要凭借外键ID去这些表里面索引就行了

5.关系数据库中的三种键（key）:PK,FK,UK
PK：primary key 主键
FK：foreign key 外键
UK：unique key 唯一约束键,UK示例如下：
在这activity_stds_lookup表的官方解释文档里 有两个UK即unique key，这说明想确定唯一表项需要STANDARD_TYPE & STANDARD_UNITS两个UK才能确定唯一的表项，只知道一个UK，返回之则可以是0到多个
Table storing details of standardised activity types and units, with permitted value ranges. 
Used to determine the correct standard_type and standard_units for a range of published types/units.
还有DK(default key 默认约束键) CK(check key 检查约束键)

6. 直接使用中的MySQL表
6.1 主表活性物表 Activity：
@activities   作为活性物数据总表
@activity_properties  信息次表 作为内嵌字典
@activity_stds_lookup  #作为解释表与Activity并列
@activity_supp_map 作为桥梁表 连接activity与activity_supp
@activity_supp 信息次表 作为内嵌字典
@ligand_eff 外键索引表
@predicted_binding_domains 信息次表，将activity与binding_site相关联


6.2 参考文献表 Assay:
@assays  #作为参考文献数据总表
@assay_class_map 桥梁表 通过assay_id关联到对应的assay_class_id,并不是所有的assay都有class
@assay_classification 作为信息次表 通过上面的class_id 去到表assay_classification寻找对应的class,再将其作为内嵌字典，存入Assay
@assay_parameters 作为信息次表 这个可以直接通过assay_id 去到表assay_parameters 内寻找对应的parameter,再将其作为内嵌字典存入Assay
@assay_type 外键的索引表

6.3 细胞cell表 Cell
@cell_dictionary #作为细胞总表

6.4 组成成分表 component
@component_sequences #作为component 组成成分数据总表
@component_synonyms 信息次表 通过component_id 进行索引，再将其作为内嵌字典
@component_go  作为桥梁表 通过component_id和go_id 将component和go_classification两表相关联
@component_domains 作为桥梁表 通过component_id和domain_id 将component和domain两表相关联
@component_class 作为桥梁表,通过component_id和protein_class_id将component与protein_classification 两表之间相关联。


6.5 小分子化合物表Compound、molecule
@molecule_dictionary 作为小分子化合物数据总表
@molecule_atc_classification 桥梁表 与下方的7.2atc_classification相关联
@molecule_frac_classification 桥梁表 与frac_classification相关联
@mole_hierarchy 多对多桥梁？？ 说明某一个molecule小分子是否有parent_molregno以及active_molregno两种属性以及对应的molecule_id
【注】：hierarchy ：等级制度
@molecule_hrac_classification 桥梁表 与 hrac_classification 相关联
@molecule_irac_classification 桥梁表 与irac_classification相关联
@compound_structural_alerts 桥梁表 与structural_alerts相关联
@molecule_synonyms 信息次表 通过molregno 索引后做成内嵌字典 
@compound_properties 信息次表 通过molregno 索引后做成内嵌字典 
@compound_records 桥梁表 将molregno与doc_id相关联 同时做信息次表索引后作为内嵌字典！！！！！！！！
@compound_structures 信息次表 通过molregno 索引后做成内嵌字典 

6.6  关联文档表 Docs
@docs 作为关联文档表 数据总表

6.7 域名表？？？Domains
@domains 作为域名数据总表


6.8 记录表record ：
@compound_records 作为record的数据总表

6.9 药物表 Drugs
drug 表有些特殊，它没有类似drug_id这样的主键以及数据总表，但是，他和record表以及molecule表关系密切，如下
@drug_indication 药物指示表 ，作桥梁表连接record以及molecule
@@indication_refs 索引表

@drug_mechanism 药物机制，药物构造表，作桥梁表关联record以及molecule
@@mechanism_refs 索引表

@drug_warning 药物警告表 作索引表，由record_id进行关联索引
@warning_refs 索引表

所以干脆，分别存三张表了,如果以后想在mongo里的molecule&compound 表里得知该化合物对应drug的关系，就直接以molecule_id为外键来这三张表里面查就行了。

6.10 metabolism 表
@metabolism 数据总表
@metabolism_refs 信息次表作为内嵌字典

6.11 靶点蛋白表target
@target_dictionary 数据总表
@target_type 信息次表，索引表 做成内嵌字典
@target_relations 桥梁表 但它是连接本身，即与A靶点蛋白有联系的其他靶点蛋白




6.12 产品表 products  
@products 数据总表
@product_patents 信息次表
@patent_use_codes 索引表做成内嵌字典
@formulations 桥梁索引表，将product与record以及molecule相关联 【注】：formulation（配方）。。









7. 间接索引的MySQL表（后续需要再补上）
@action_type
@atc_classification 多用于外表索引
@activity_smid 多用于外表索引 
@binding_sites 多用于外表索引 主键为site_id
@bio_component_sequences 多用于外表索引，主键为component_id
@bioassay_ontology 多用于外表索引，主键为bao_id

@biotherapeutic_components 作用于之后的小分子化合物表，与molregno以及component_id 相关联 
@biotherapeutics 作用与之后的小分子化合物表，与molregno相关联。

chembl_id_lookup 多用于外表索引
@usan_stems

@各种......lookup
'''
