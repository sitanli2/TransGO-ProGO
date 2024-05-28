from pymongo import MongoClient
import pandas as pd



def MongoConnection():
    mongoClient = MongoClient('mongodb://localhost:27017/')
    database = mongoClient.get_database('dtmr_dev')

    # collection = dtmr_dev['uniprot_Target_Info_test']

    collection = database.get_collection('uniprot_Target_Info_test')
    return collection

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

def Gen_GO_labelTXT():
    """用来给MonGoDB中的每一个条目都加上mf,bp,cc"""
    collection = MongoConnection()
    data = pd.read_csv('G:/fkfk/AlphaFold_refine/Protein_spider/UniProt_Scrapy/uniprot_id.csv', engine='python')  # sep='/n',
    uniprot_id_list, uniprot_name_list = return_id_name_list(data)
    FullSize = int(len(uniprot_id_list)) # TestSize = int(input("输入你想要在Mongo中更新（加上mf,bp,cc）的条目数"))
    testcount = 0
    for UPID in uniprot_id_list:
        errorcode = 0
        try:
            query_result = collection.find_one({"UPID": UPID})["Go_info"]  # 从MonGoDB中索引到一个条目的 Go_info,
            # 这里需要注意一个问题，有可能出现errorID的情况（网络接口返回文件损坏如UPID=A6H8Y1），详情见爬虫部分，
            # 我把这5个errorID单独存起来了，所以当遍历到它们的ID时，靠上面这行代码检索将会出错，应为他们在Mongo中为null
            # 所以在这里可以给这个查询加一个判错，当遍历到errorID（查询失败时），自动跳过。
            print("查询结果:", query_result, '\n', type(query_result))
        except Exception as error:
            print('错误类型是', error.__class__.__name__)
            print('错误明细是', error)
            errorcode = 1


        MFlist = []
        BPlist = []
        CClist = []
        if (len(query_result) > 0 and errorcode == 0):  # 不为空列表，即拥有Go_info（一般来说肯定都是有的，除非爬虫出问题了导致上面的查询出错）
            for item in query_result:
                Golabel = item["id"]
                property = item["properties"]
                for prop in property:
                    if (prop["key"] == "GoTerm"):
                        if (str(prop["value"])[0] == "C"):  # 说明这个Golabel是一个CC 标签
                            CClist.append(Golabel)
                        elif (str(prop["value"])[0] == "F"):  # 说明这个Golabel是一个MF 标签
                            MFlist.append(Golabel)
                        elif (str(prop["value"])[0] == "P"):  # 说明这个Golabel是一个BP 标签
                            BPlist.append(Golabel)
                    break  # 这个循环主要是为了遍历整个 properties 列表，拿到了我们想要的GoTerm后就要跳出这个循环
        else:
            print("UPID=" + UPID + "的MonGoData没有Go标签条目")
            pass #存入空列表 mf,bp,cc

        # 通过上面的一系列操作后，我们拿到了这条UPID对应的 所有Go_label,并将它们分别存进了对应的三个category的列表中
        record = {}
        record["mf"] = MFlist
        record["bp"] = BPlist
        record["cc"] = CClist

        # 更新文档
        update_result = collection.update_one({"UPID": UPID}, {"$set": record})  # 给每个条目都加上record

        # 打印更新后的文档
        if update_result.modified_count > 0:
            print("更新成功,更新条目：", record)

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

        testcount += 1  # 用来标记测试数据的条数
        if (testcount == FullSize):
            break

# def Gen_GO_labelTXT_usecursor():
#     TestSize = int(input("输入你想要在Mongo中更新（加上mf,bp,cc）的条目数"))
#     collection = MongoConnection()
#     cursor = collection.find()  # 获取一个游标（cursor），然后逐个迭代游标以按顺序遍历MongoDB中的每个条目访问每个文档。
#     for document in cursor:



def GetFullIDlist():
    """
    为了避免重复 根据本地的Database1 的 pdb_mapping_seq.txt文件获取所有的七千多条UPID 和 PID
    :return:FullUPIDlist,FullPIDlist
    """
    path = "/AlphaFold_refine/biLSTM+GCN/data_collect/pdb_mapping_seq.txt"
    FullUPIDlist = []
    FullPIDlist = []
    with open(path,"r") as file:
        for line in file:
            if '>' in line:
                pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                PID = pid_chain_uid.split("_")[0]
                UPID = pid_chain_uid.split("_")[1]
                FullUPIDlist.append(UPID)
                FullPIDlist.append(PID)
                # print("test data UPID =",UPId,"test data PID =",PID)
                # break
        #print("test data UPID len=", len(FullUPIDlist), "test data PID len=", len(FullPIDlist))
    return FullUPIDlist,FullPIDlist


def getAlphaEntryIDFromMongo(UPID):
    collection = MongoConnection()
    try:
        query_result = collection.find_one({"UPID": UPID})["Structure_AlphaFold"]["IDENTIFIER_entryId"]  # 从MonGoDB中通过UPID索引到其对应条目的 Structure_AlphaFold 嵌套字典
        print(UPID+"的IDENTIFIER_entryId 查询结果:", query_result,'\n返回类型：', type(query_result))
        entryID = query_result
    except:
        print("查询不到与UPID=",UPID,"对应的AlphaFold Entry条目，没有与之对应的AlphaFold预测结构")
        entryID = 'null'
    return entryID


def LabelhaveMf_andAll_InMongo():
    """
    此函数用来访问Mongo数据库中爬虫爬取的数据，将他们对应的6000条数据的UPID保存下来生成Database2的 Intact_label.txt
    这6000条数据应该满足以下筛选条件：
    - 通过 GetFullIDlist() 获得Database1 的数据UPID,Database2筛选的数据不应该与Database1的数据发生重复
    - 由于ESM-1b 能处理的氨基酸序列最大长度为1022个氨基酸+首token+尾token  所以所有蛋白质序列长度超过1022的都不行，否则就需要：（为了省事，我就直接排除序列长度超过1022的蛋白质了）
        ESM - 1b 模型在不超过1024个残基限制（这1024个residue还包括了起始和终止的token,所以真正的训练残基数不能超过1022个）的条件下进行训练
        所以对于长度大于1022的蛋白质序列，比如说序列长度为n，我们将序列分为 n / 1022 个长度为1022的块，除去最后一个长度为 n % 1022的块（去除余数），每一个块（1022长的氨基酸序列）都生成一个嵌入（embedding），
        最后将所有块的嵌入连接（concate 拼接，连接）起来作为这个序列长度超过1022的蛋白质的嵌入。
    - 最后按照mf,bp,cc都要有的条件来筛选出最终的6000条数据作为Database2

    :return:
    """
    IntactLabelID_list = []
    collection = MongoConnection()
    cursor = collection.find() #获取一个游标（cursor），然后逐个迭代游标以按顺序遍历MongoDB中的每个条目访问每个文档。
    FullUPIDlist,FullPIDlist = GetFullIDlist()
    testcount = int(input("你需要从数据库中筛选出几条ID呢？"))
    accomplish = 0 #用来标志是否完成了设定的目标,i.e:testcount == len(IntactLabelID_list)
    for document in cursor:
        print(document,"\n",type(document))
        if (document["UPID"] in FullUPIDlist):
            print("该protein条目已经在Database1中存在了，跳过")
            continue
        errorID = ['A6H8Y1','P08579','Q06190','Q969Q6','Q99437','Q9Y5P8'] #这六个ID是在爬虫阶段产生错误的ID无法解析他们的数据接口返回值，需要单独用beautifulsoup或其他方法直接解析他们的网页HTML拿数据
        if (document["UPID"] in errorID):
            print("错误ID",document["UPID"])
            continue
        if ('inactive' in document) :
            # 数据库中还存在 未激活状态的数据比如：
            # {'_id': ObjectId('662deb4021712c0f197615b9'), 'UPID': 'Q08AF8', 'inactive': 'True', 'inactiveReasonType': ['DEMERGED'], 'bp': [], 'cc': [], 'mf': []}
            print("该数据在UniprotKB中未激活，inactiveReasonType = ",document['inactiveReasonType'])
            continue
        if (getAlphaEntryIDFromMongo(document["UPID"]) == 'null'): #在后续的测试中发现，有些蛋白质数据是没有AlphaFold预测结构的（比如UPID =A2VEC9、P22352..... ），这会导致数据缩水，所以要提前排除
            print("该UPID=",document["UPID"],"的蛋白质没有与之对应的AlphaFold结构")
            continue
            # break
        if (len(str(document["Sequence"])) >= 1022): #该蛋白质序列长度超过了Esm能处理的极限，跳过该蛋白
            continue

        if ("mf" in document and "bp" in document and "cc" in document): #先检验一下，该条目下这三个文档是否成功通过上面的 Gen_GO_labelTXT()函数 添加
            #if ("Structure_AlphaFold" in document): #按理来说这些数据都会有AlphaFold entry的
            if(len(document["mf"]) >= 2 and len(document["bp"]) > 1 and len(document["cc"]) > 1):
                #这一步就是筛选看这个条目是否三个标签都存在，当然还可以更严格一点如要求三个标签的个数要大于某个阈值,我对mf标签比较严格只选取mf标签大于等于2个的条目
                IntactLabelID_list.append(document["UPID"])
        else:
            print("该条目还没通过Gen_GO_labelTXT()生成mf,bp,cc索引")
        if (testcount == len(IntactLabelID_list)):
            accomplish = 1 #完成后置1
            break #这里我做了一个测试 没有break让他一直执行下去，最后得到的IntactLabelID_list 中有6459条数据，说明在爬下来的2万多条数据里面共有6459条数据满足上面的筛选条件
    if (accomplish == 1):
        print("value of the list:", IntactLabelID_list)
        print("成功筛选出指定",len(IntactLabelID_list),"条数据")
        return IntactLabelID_list
    else:
        print("value of the list:", IntactLabelID_list)
        print("爬虫基准数据库无法筛选出你想要的条数=",testcount,"最多能筛选出的条数为=",len(IntactLabelID_list),"\n")
        return IntactLabelID_list

# def getAlphaEntryIDFromMongo(UPID):
#     collection = MongoConnection()
#     try:
#         query_result = collection.find_one({"UPID": UPID})["Structure_AlphaFold"]["IDENTIFIER_entryId"]  # 从MonGoDB中通过UPID索引到其对应条目的 Structure_AlphaFold 嵌套字典
#         print(UPID+"的IDENTIFIER_entryId 查询结果:", query_result,'\n返回类型：', type(query_result))
#         entryID = query_result
#     except:
#         print("查询不到与UPID=",UPID,"对应的AlphaFold Entry条目，没有与之对应的AlphaFold预测结构")
#         entryID = 'null'
#     return entryID




def GenIntact_lebelTXT():
    filepath = "G:/fkfk/AlphaFold_refine/UltraDataset/Database2_Uniprot+AlphaFold/Intact_label.txt" #最终的生成路径
    # 将列表中的元素逐行写入到文本文件
    IntactLabelIDlist = LabelhaveMf_andAll_InMongo()
    with open(filepath, 'w') as file:
        for item in IntactLabelIDlist:
            AlphaFold_entryID = getAlphaEntryIDFromMongo(item)
            if(AlphaFold_entryID == 'null'):
                continue #这条蛋白质没有合适的AlphaFold预测结构
            full_item = str(AlphaFold_entryID) +'_'+ str(item) #拼接起来，做为最终的ID
            file.write(full_item + '\n')



# def LabelhaveMf_andAll_Indisk(getnum):
#     """
#     此函数用来从data_collect 那个Go_label原件中筛选出所有有MF标签的IDlist 即 mfIDlist
#     随后再筛选出 BP,MF，CC 三个标签都有的IDlist 即 IntactLabelIDlist
#     :parameter:getnum IntactLabelIDlist需要几条 ，即拥有完整bp,cc,mf 的ID
#     :return: mfIDlist IntactLabelIDlist
#     """
#     IntactLabelIDlist = [] #用来存放mf,bp,cc三个标签都有的ID
#     mfIDlist = [] #用来存放有mf标签的这些ID
#     with open('../../data_collect/Go_label.txt', 'r') as f: #data_collect 下的Go_label 原件
#         for line in f:
#             if '>' in line: #说明这一行是ID 如 >4WKG-A_P77398，也作为新的一个开始需要把所有Flag置为0
#                 mfflag = 0
#                 bpflag = 0
#                 ccflag = 0
#                 pdb_chain_uid = line[1:].strip('\n')
#             elif 'mf' in line: #当前行是mf标签行
#                 if 'mf:' == line.strip('\n'): #说明这一整行 只有一个 ‘mf:’ ,即mf标签为空
#                     # NoMfIDlist.append(pdb_chain_uid)
#                     pass #当然也可以向上面一样存一下这个ID
#                 else:
#                     mfflag= 1 #将flag置为1
#                     mfIDlist.append(pdb_chain_uid)
#             elif 'bp' in line: #当前行是bp标签
#                 if 'bp:' != line.strip('\n'): #看这一行bp标签是否为空，不为空将对应的flag置为一
#                     bpflag = 1
#             elif 'cc' in line: #当前行是cc标签,经过几次循环后下一行有 '>' 这个时候就要判断一下几个flag了
#                 if 'cc:' != line.strip('\n'): #看这一行cc标签是否为空，不为空将对应的flag置为一
#                     ccflag = 1
#                 if (mfflag == 1 and bpflag == 1 and ccflag == 1): # 三个标签都有
#                     IntactLabelIDlist.append(pdb_chain_uid)
#             if(len(IntactLabelIDlist) >= getnum):
#                 break
#         return mfIDlist,IntactLabelIDlist
#
#
#
#
#
# def GenIntact_lebelTXT(IntactLabelIDlist):
#     filepath = "Intact_label.txt"
#     # 将列表中的元素逐行写入到文本文件
#     with open(filepath, 'w') as file:
#         for item in IntactLabelIDlist:
#             file.write(item + '\n')



if __name__ == '__main__':
    # Gen_GO_labelTXT() #给Mongo中的数据生成mf,bp,cc索引，起初只给Mongo数据库中前5000条数据做了这个更新
    # LabelhaveMf_andAll_InMongo() #返回筛选出来的ID列表,并将其存为同级目录下的的intact_label.txt
    # GenIntact_lebelTXT() #生成最终筛选的 AlphaFoldentryID与UPID的组合的Intact_label.txt
    pass






