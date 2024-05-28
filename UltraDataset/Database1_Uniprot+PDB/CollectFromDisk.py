
"""
这个脚本主要是用来精炼原始PDB-B数据集的 (从之前的7198条的数据集缩减（精炼，只取拥有全标签的数据）为3000条)
解释一下具体的流程
- 读取本地的PDB数据（data_collect）中的 Go_label.txt ,通过LabelhaveMf_andAll 函数筛选出3000条UPID,并将其存为csv文件保存为 Intact_Label.txt
- 通过上面筛选出来的Intact_Label.csv 对原Go_label.txt 进行精炼生成精炼后的Go_label.txt
- 接下来使用 CopyContactMapFromFilePath 将所有intact id的contact map从本地复制到当前的contact目录下
- 生成精炼后的 pdb_mapping_seq.txt 文件
- 使用 getnorepeat_mffuclist 和 generate_mf_labelTXT函数对mf（,bp,cc也一样）生成专属他的mf_label.txt

"""

def LabelhaveMf_andAll(getnum):
    """
    此函数用来从data_collect 那个Go_label原件中筛选出所有有MF标签的IDlist 即 mfIDlist
    随后再筛选出 BP,MF，CC 三个标签都有的IDlist 即 IntactLabelIDlist
    :parameter:getnum IntactLabelIDlist需要几条 ，即拥有完整bp,cc,mf 的ID
    :return: mfIDlist IntactLabelIDlist
    """
    IntactLabelIDlist = [] #用来存放mf,bp,cc三个标签都有的ID
    mfIDlist = [] #用来存放有mf标签的这些ID
    with open('../../biLSTM+GCN/data_collect/Go_label.txt', 'r') as f: #data_collect 下的Go_label 原件
        for line in f:
            if '>' in line: #说明这一行是ID 如 >4WKG-A_P77398，也作为新的一个开始需要把所有Flag置为0
                mfflag = 0
                bpflag = 0
                ccflag = 0
                pdb_chain_uid = line[1:].strip('\n')
            elif 'mf' in line: #当前行是mf标签行
                if 'mf:' == line.strip('\n'): #说明这一整行 只有一个 ‘mf:’ ,即mf标签为空
                    # NoMfIDlist.append(pdb_chain_uid)
                    pass #当然也可以向上面一样存一下这个ID
                else:
                    mfflag= 1 #将flag置为1
                    mfIDlist.append(pdb_chain_uid)
            elif 'bp' in line: #当前行是bp标签
                if 'bp:' != line.strip('\n'): #看这一行bp标签是否为空，不为空将对应的flag置为一
                    bpflag = 1
            elif 'cc' in line: #当前行是cc标签,经过几次循环后下一行有 '>' 这个时候就要判断一下几个flag了
                if 'cc:' != line.strip('\n'): #看这一行cc标签是否为空，不为空将对应的flag置为一
                    ccflag = 1
                if (mfflag == 1 and bpflag == 1 and ccflag == 1): # 三个标签都有
                    IntactLabelIDlist.append(pdb_chain_uid)
            if(len(IntactLabelIDlist) >= getnum):
                break
        return mfIDlist,IntactLabelIDlist

def GenIntact_lebelTXT(IntactLabelIDlist):
    filepath = "Intact_label.txt"
    # 将列表中的元素逐行写入到文本文件
    with open(filepath, 'w') as file:
        for item in IntactLabelIDlist:
            file.write(item + '\n')


def GetNewGo_labelTXT(IntactLabel_IDlist):
    """在UltraDataset下面生成精炼后的Go_label.txt"""

    input_file_path = '../../biLSTM+GCN/data_collect/Go_label.txt'
    output_file_path = 'Go_label.txt'
    start_writing = False  # 记录是否开始写入的标志
    lines_to_write = 0  # 这个参数表示要写几行

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        count = len(IntactLabel_IDlist)
        for line in input_file:
            if (">" in line and line[1:].strip('\n') in IntactLabel_IDlist): # 指定这一行的内容作为开始写入的标志 line[1:].strip('\n')就是pdb_chain_uid
                """说明这一行是ID行 如 >4WKG-A_P77398,我们要看它是否属于IntactLabel_IDlist 属于的话，则它和它的三个标签就写入新的Go_label.txt中"""
                start_writing = True
                lines_to_write = 3  # 写入自己这一行（ID行）和接下来的3行（mf,bp,cc）
                count -= 1 #写入了一条ID和他对应的label

            if start_writing and lines_to_write >= 0:
                output_file.write(line)
                lines_to_write -= 1

            if(count <= 0 and lines_to_write < 0): #写入了len(IntactLabel_IDlist)条后（写完了）就不需要再循环了
                break

def getNewseqMapping():
    IDlist = readIntact_label()
    print(IDlist, "\n", len(IDlist))
    filepath = "/AlphaFold_refine/biLSTM+GCN/data_collect/pdb_mapping_seq.txt"
    output_file_path = "G:/fkfk/AlphaFold_refine/UltraDataset/Database1_Uniprot+PDB/pdb_mapping_seq.txt"
    with open(filepath, 'r') as inputfile , open(output_file_path, 'w') as output_file:
        count = len(IDlist)
        for line in inputfile:
            ID = line.split()[0].strip('>') #拿到ID 如 4WKG-A_P77398
            # print(ID)
            # break
            if ID in IDlist: #若这个ID属于精炼ID则这一整行都要写入
                output_file.write(line)
                count -= 1
            if(count <= 0):
                break


def getnorepeat_mffuclist():
    """
    此函数用来返回UltraDataset下面的 Golabel.txt文件中所有的3000条ID中所含的所有不重复的mf标签，
    然后再将其通过下面的函数 generate_mf_labelTXT 写入UltraDataset下面的/Database1_Uniprot+PDB/mf/mf_label.txt
    :return:
    """
    mf_func = []
    mf_func_norepeat = []
    with open('Go_label.txt', 'r') as f: #UltraDataset下面的 精炼过后的 Golabel.txt

        for line in f:
            if '>' in line:
                pdb_chain_uid = line[1:].strip('\n')
            elif 'mf' in line:
                if 'mf:' == line.strip('\n'): #mf 标签为空
                    continue
                else:
                    mf_func_list = line[3:].strip().split('\t')
                    #print(mf_func_list,"\n")
                    for i in mf_func_list:
                        if i not in mf_func_norepeat:
                            mf_func_norepeat.append(i)
    return mf_func_norepeat

def generate_mf_labelTXT(mf_func_norepeat_file):
    """
    写入 未重复的 mf_label
    :param mf_func_norepeat_file:
    :return:
    """
    with open('./mf/mf_label.txt', 'w') as file:
        for item in mf_func_norepeat_file:
            file.write("%s\n" % item)


def CopyContactMapFromFilePath(path,filename_list,destination_folder):
    """

    :param path: 就是放所有contact map 的绝对路径 G:/fkfk/AlphaFold_refine/data_collect/contact_map_dense_pdb_8.0/
    :param filenames: 需要复制到新文件夹（testData）下的contact map的索引（文件名）列表,比如说filenames = [1A0A-A_P07270.npy,.....]
    :parameter destination_folder: testData下面放contact map 的文件夹
    :return:
    """
    import os
    import shutil
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in filename_list:
        filename = filename + ".npy"
        source_path = os.path.join(path, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy(source_path, destination_path)
        print(f"文件 {filename} 已复制到 {destination_folder}")

def readIntact_label():
    IDList = []
    with open("Intact_label.txt", 'r') as file:
        for line in file:
            IDList.append(line.strip())
    return IDList


def SplitIDList():
    IDlist = readIntact_label()
    """分割比例设定为 9：0.5：0.5"""
    total_len = len(IDlist) #3000
    first_len = int(total_len * 0.9)
    second_len = int(total_len * 0.05)

    first_list = IDlist[:first_len] #0.9*3000=2700 条数据作为train data
    second_list = IDlist[first_len:first_len + second_len] #vlidata data
    third_list = IDlist[first_len + second_len:] #test data
    print("训练集长度:", len(first_list))
    print("验证集长度:", len(second_list))
    print("测试集长度:", len(third_list))

    return first_list,second_list,third_list

def FromListToTXTFile(list,file_path):
    with open(file_path, 'w') as file:
        for item in list:
            file.write(str(item) + '\n')

def GetFullIDlist():
    """
    根据本地的Database1 的 pdb_mapping_seq.txt文件获取所有的七千多条UPID 和 PID
    :return:
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
        print("test data UPID len=", len(FullUPIDlist), "test data PID len=", len(FullPIDlist))
    return FullUPIDlist,FullPIDlist



if __name__ == '__main__':
    # mf_func_norepeat_file = getnorepeat_mffuclist()
    # print("no repeat func list:", mf_func_norepeat_file)
    #generate_mf_labelTXT(mf_func_norepeat_file)

    """先生成精炼的Intact_label.csv 这个文件中的每个id都是"""
    # i = input("输入你要的拥有完整label 的ID条数=") #暂定为3000条，即IntactIDlist
    # mfIDlist,IntactLabelIDlist = LabelhaveMf_andAll(int(i)) #mfIDlist 是不缺少mf标签的ID,IntactIDlist 是mf,bp,cc 都不缺少的ID
    # GenIntact_lebelTXT(IntactLabelIDlist)

    """读取上面生成的Intact_label.txt 生成最终的精炼版Go_label.txt"""
    # IntactLabelList = []
    # with open("Intact_label.txt", 'r') as file:
    #     for line in file:
    #         IntactLabelList.append(line.strip())
    #     # print(IntactLabelList,"\n",len(IntactLabelList))
    # testIDlist = ["4WKG-A_P77398","1E0T-A_P0AD61","3BHO-A_O43809","1B48-A_P24472"]
    # GetNewGo_labelTXT(IntactLabelList) # testIDlist , IntactLabelList

    """从本地复制对应的contact map 到新的精炼后contact目录下面"""
    # ContactFileList = []
    # with open("Intact_label.txt", 'r') as file:
    #     for line in file:
    #         ContactFileList.append(line.strip())
    #     #print(ContactFileList, "\n", len(ContactFileList))
    # path = "G:/fkfk/AlphaFold_refine/data_collect/contact_map_dense_pdb_8.0/"
    # destination_folder = "./contact_map_dense_pdb_8.0"
    # CopyContactMapFromFilePath(path,ContactFileList,destination_folder)

    """验证Contact map 是否都存进来了"""
    # import os
    #
    # files_list = os.listdir("./contact_map_dense_pdb_8.0")# 获取目标路径下的所有文件列表
    # total_files = len(files_list) # 统计文件数量
    # print(f"目标路径下共有 {total_files} 个文件。") # 打印文件数量,结果为3000 验证成功

    """生成 pdb_mapping_seq.txt"""
    #getNewseqMapping()

    """生成mf文件夹中的mf_label.txt"""
    # mf_func_norepeat = getnorepeat_mffuclist() #返回精炼后的Golabel.txt中所有的mf标签
    # print("no repeat func list:", mf_func_norepeat,"\nnumber of the no repeat mf label=",len(mf_func_norepeat)) #从精炼前的486 变成了482
    # generate_mf_labelTXT(mf_func_norepeat)

    """读取Intact_label.txt文件，将精炼后的3000条ID按比例（9:0.5:0.5）分为train_data.txt, valid_data.txt, test_data.txt"""
    #train_list,vlidate_list,test_list= SplitIDList() # 2700条，150条，150条
    # FromListToTXTFile(train_list,"./mf/train_data.txt")
    # FromListToTXTFile(vlidate_list,"./mf/valid_data.txt")
    # FromListToTXTFile(test_list,"./mf/test_data.txt")

    """为了与后面的Uniprot+AlphaFold 数据库即database2 做区分避免重复数据，这里要得到这所有七千多条数据的UPID和PID"""
    # FullUPIDlist,FullPIDlist= GetFullIDlist()
    # print(FullUPIDlist,"\n",FullPIDlist)
    """接下来在PDB_Scrapy 下面生成uniprot_id.txt 对照Uniprot_Scrapy 下面的uniprot_id.csv"""
    # with open('G:/fkfk/AlphaFold_refine/Protein_spider/PDB_Scrapy/uniprot_id.txt', 'w') as file:
    #     for item in FullUPIDlist:
    #         file.write("%s\n" % item)
    pass



