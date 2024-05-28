from pymongo import MongoClient
import pandas as pd

def return_id_name_list(data):
    i = 0#行指针
    j = 0#列指针(这个csv读取后生成的dataframe文件只有两列，一列id一列name)
    uniprot_id_list = []
    uniprot_name_list = []
    hang_len = data.shape[0]
    lie_len = data.shape[1]
    while(i <= hang_len-1 and j <= lie_len-1):
        uniprot_id_list.append(data.loc[i][j])
        i += 1#换行
        j += 1#换到第二列得到name
        uniprot_name_list.append(data.loc)
        j = 0 #列换回到第一列id列
    return uniprot_id_list,uniprot_name_list

def MongoConnection():
    mongoClient = MongoClient('mongodb://localhost:27017/')
    database = mongoClient.get_database('dtmr_dev')

    # collection = dtmr_dev['uniprot_Target_Info_test']

    collection = database.get_collection('uniprot_Target_Info_test')
    return collection



def Gen_GO_labelTXT(MaxSize):
    """用来给MonGoDB中的每一个条目都加上mf,bp,cc"""

    collection = MongoConnection()
    data = pd.read_csv('../uniprot_id.csv', engine='python')  # sep='/n',
    uniprot_id_list, uniprot_name_list = return_id_name_list(data)
    testcount = 0
    for UPID in uniprot_id_list:
        query_result = collection.find_one({"UPID": UPID})["Go_info"] #从MonGoDB中索引到一个条目的 Go_info
        print("查询结果:", query_result,'\n',type(query_result))

        MFlist = []
        BPlist = []
        CClist = []
        if (len(query_result)>0): #不为空列表，即拥有Go_info（一般来说肯定都是有的，除非爬虫出问题了）
            for item in query_result:
                Golabel = item["id"]
                property = item["properties"]
                for prop in property:
                    if(prop["key"] == "GoTerm"):
                        if(str(prop["value"])[0] == "C"): #说明这个Golabel是一个CC 标签
                            CClist.append(Golabel)
                        elif(str(prop["value"])[0] == "F"): #说明这个Golabel是一个MF 标签
                            MFlist.append(Golabel)
                        elif(str(prop["value"])[0] == "P"): #说明这个Golabel是一个BP 标签
                            BPlist.append(Golabel)
                    break #这个循环主要是为了遍历整个 properties 列表，拿到了我们想要的GoTerm后就要跳出这个循环
        else:
            print("UPID="+UPID+"的MonGoData没有Go标签条目")
            pass

        #通过上面的一系列操作后，我们拿到了这条UPID对应的 所有Go_label,并将它们分别存进了对应的三个category的列表中
        record = {}
        record["mf"] = MFlist
        record["bp"] = BPlist
        record["cc"] = CClist

        # 更新文档
        update_result = collection.update_one({"UPID": UPID}, {"$set": record}) #给每个条目都加上record

        # 打印更新后的文档
        if update_result.modified_count > 0:
            print("更新成功,更新条目：",record)

        else:
            print("未找到对应的文档")

        """
        我们最后的目的是生成最后的Go_label.txt 文件，element如下所示：
        
        >1E0T-A_P0AD61
        mf:GO:0016772	GO:0016773	GO:0016301	GO:0097367	GO:0042802	GO:0032559	GO:0000287	GO:0017076	GO:0032555	GO:0005524	GO:0035639	GO:0030554	GO:0032553	
        bp:GO:0006950	GO:0009135	GO:0046939	GO:0016052	GO:0019637	GO:0009185	GO:0051259	GO:0006163	GO:0065003	GO:0051262	GO:0019693	GO:0009179	GO:0043436	GO:0005975	GO:0006165	GO:0016310	GO:0006757	GO:0032787	GO:0009117	GO:0071840	GO:0006091	GO:0022607	GO:1901575	GO:0009056	GO:0006753	GO:0006082	GO:0009266	GO:0055086	GO:0009259	GO:0009628	GO:0046031	GO:0044085	GO:0016043	GO:0009408	GO:0006096	GO:0006796	GO:0051260	GO:0043933	GO:0009150	GO:0009132	GO:0006793	GO:0072521	GO:1901135	GO:0019752	GO:0051289	GO:0006090	GO:0046034	
        cc:GO:0005829
        以测试数据 即：UPID=A0A087X1C5 为例：
        {'mf': ['GO:0070330', 'GO:0020037', 'GO:0005506', 'GO:0016712'], 'bp': ['GO:0019369', 'GO:0042178', 'GO:0006805'], 'cc': ['GO:0005737', 'GO:0043231', 'GO:0016020', 'GO:0005739']}
        
        于是它在Go_label.txt中：
        >A0A087X1C5 （AF-A0A087X1C5-F1 这个ID在MongoDB中的Structure_AlphaFold条目下可以找到,可以通过这个ID爬取AlphaFold的PDB文件）
        mf:GO:0070330   GO:0020037  GO:0005506  GO:0016712
        bp:GO:0019369   GO:0042178  GO:0006805
        cc:GO:0005737   GO:0043231  GO:0016020  GO:0005739
        """

        testcount  += 1 #用来标记测试数据的条数
        if(testcount == MaxSize):
            break

if __name__ == '__main__':
    Maxsize = 1
    Gen_GO_labelTXT(Maxsize)