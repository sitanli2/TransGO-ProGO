import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
from featureGen import embed_sequence
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import esm


"""使用小批量数据（如testData时）测试时使用该工具包"""

def loading_contact_map(data_source,threshold):
    contact_map_list = os.listdir('G:/fkfk/AlphaFold_refine/UltraDataset/testData/contact_map_dense_pdb_'+str(threshold))
    dict_contact_map = {}
    for item in tqdm(contact_map_list):
        if data_source == 'pdb':
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('G:/fkfk/AlphaFold_refine/UltraDataset/testData/contact_map_dense_pdb_'+str(threshold)+'/'+item)
        elif data_source == 'alpha':
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('G:/fkfk/AlphaFold_refine/UltraDataset/testData/contact_map_dense_alphafold_'+str(threshold)+'/'+item)
    return dict_contact_map



def loading_seq(data_source):
    dict_seq = {}
    if data_source == 'pdb':
        with open('G:/fkfk/AlphaFold_refine/UltraDataset/testData/pdb_mapping_seq.txt','r') as f:
            for line in tqdm(f,total=10): #精炼后的数据只有3000条了
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb

    elif data_source == 'alpha':
        with open('../biLSTM+GCN/data_collect/alpha_mapping_seq.txt') as f:
            for line in tqdm(f,total=7198):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb
    return dict_seq

def get_label(func):
    if func == 'mf':
        mf_dict = {}
        mf_func = []
        mf_label_dict = {}
        with open('G:/fkfk/AlphaFold_refine/UltraDataset/testData/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'mf' in line:
                    if 'mf:' == line.strip('\n'):
                        continue
                    else:
                        mf_func_list = line[3:].strip().split('\t')
                        mf_dict[pdb_chain_uid] = mf_func_list
        with open('G:/fkfk/AlphaFold_refine/UltraDataset/testData/mf/mf_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                mf_func.append(line)
        for i in mf_dict.keys():
            label = np.zeros(len(mf_func))
            for j in mf_dict[i]:
                if j in mf_func:
                    index = mf_func.index(j)
                    label[index] = 1
            mf_label_dict[i] = label
        return mf_label_dict,len(mf_func)
    elif func == 'bp':
        bp_dict = {}
        bp_func = []
        bp_label_dict = {}
        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'bp' in line:
                    if 'bp:' == line.strip('\n'):
                        continue
                    else:
                        bp_func_list = line[3:].strip().split('\t')
                        bp_dict[pdb_chain_uid] = bp_func_list
        with open('./data_collect/bp/bp_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                bp_func.append(line)
        for i in bp_dict.keys():
            label = np.zeros(len(bp_func))
            for j in bp_dict[i]:
                if j in bp_func:
                    index = bp_func.index(j)
                    label[index] = 1
            bp_label_dict[i] = label
        return bp_label_dict,len(bp_func)
    elif func == 'cc':
        cc_dict = {}
        cc_func = []
        cc_label_dict = {}
        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'cc' in line:
                    if 'cc:' == line.strip('\n'):
                        continue
                    else:
                        cc_func_list = line[3:].strip().split('\t')
                        cc_dict[pdb_chain_uid] = cc_func_list
        with open('./data_collect/cc/cc_label.txt', 'r') as f:
            for line in f:
                line = line.strip('\n')
                cc_func.append(line)
        for i in cc_dict.keys():
            label = np.zeros(len(cc_func))
            for j in cc_dict[i]:
                if j in cc_func:
                    index = cc_func.index(j)
                    label[index] = 1
            cc_label_dict[i] = label
        return cc_label_dict, len(cc_func)
def loading_item_data(func,name):
    item = []
    with open('G:/fkfk/AlphaFold_refine/UltraDataset/testData/'+func+'/'+name+'_data.txt','r') as f:
        for line in f:
            line = line.strip('\n')
            item.append(line)
    return item



class PFPDataset(InMemoryDataset): # 定义子类PFPDataset继承InMemoryDataset
    def __init__(self,model_feature=None,train_data_X=None,train_contactmap_X=None,train_feature_matrix_X=None,train_data_Y=None,root='/tmp',transform=None,pre_transform=None,device = None):
        super(PFPDataset,self).__init__(root,transform,pre_transform)
        #self.dir=dir
        self.device = device
        self.model_feature = model_feature
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y
        self.X_contactmap_list = train_contactmap_X
        self.X_feature_matrix_list = train_feature_matrix_X
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def __len__(self):
        return int(len(self.X_data_list))

    def __getitem__(self, idx):
        """
        在 Python 中，双下划线 __ 用于标识特殊方法或属性，这些方法和属性在类中有特殊的含义和用途。
        __getitem__ 是 Python 中的一个特殊方法，用于实现索引访问操作，即通过类似 obj[key] 的方式来获取对象的元素。

        根据给定索引 idx 返回数据集中的一个样本。它首先从传入的索引 idx 中获取接触图、标签和特征矩阵，然后将特征矩阵转换为特征表示，
        最后将所有这些数据打包成一个 GCNData_mol 对象返回，其实就是将初始数据做成geometric的输入格式DATA.Data:
        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])
        """

        contact_map = self.X_contactmap_list[self.X_data_list[idx]]
        label = self.Y_data_list[self.X_data_list[idx]]
        seq = self.X_feature_matrix_list[self.X_data_list[idx]]
        seq_bytes = bytes(seq,encoding='utf-8')

        feature_matrix = embed_sequence(self.model_feature,seq_bytes,use_cuda=True,device = self.device)
        #超级牛逼，靠这个Model_feature 提取特征做成特征矩阵



        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])


        #embedding = torch.Tensor([embedding])


        return GCNData_mol


class ESMDataset(InMemoryDataset): # 定义子类ESMDataset继承InMemoryDataset
    def __init__(self,train_data_X=None,train_contactmap_X=None,train_feature_matrix_X=None,train_data_Y=None,root='/tmp',transform=None,pre_transform=None,device = None):
        super(ESMDataset,self).__init__(root,transform,pre_transform)
        #self.dir=dir
        self.device = device
        #self.model_feature = model_feature
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y
        self.X_contactmap_list = train_contactmap_X
        self.X_feature_matrix_list = train_feature_matrix_X
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def __len__(self):
        return int(len(self.X_data_list))

    def __getitem__(self, idx):
        """
        在 Python 中，双下划线 __ 用于标识特殊方法或属性，这些方法和属性在类中有特殊的含义和用途。
        __getitem__ 是 Python 中的一个特殊方法，用于实现索引访问操作，即通过类似 obj[key] 的方式来获取对象的元素。

        根据给定索引 idx 返回数据集中的一个样本。它首先从传入的索引 idx 中获取接触图、标签和特征矩阵，然后将特征矩阵转换为特征表示，
        最后将所有这些数据打包成一个 GCNData_mol 对象返回，其实就是将初始数据做成geometric的输入格式DATA.Data:
        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])
        """

        contact_map = self.X_contactmap_list[self.X_data_list[idx]]
        label = self.Y_data_list[self.X_data_list[idx]]
        seq = self.X_feature_matrix_list[self.X_data_list[idx]] #索引到的就是train_feature_matrix_X 即该条蛋白质的氨基酸序列

        seq_bytes = bytes(seq,encoding='utf-8')

        #print("seq=",seq) # 看看这个seq到底是啥

        #feature_matrix = embed_sequence(self.model_feature,seq_bytes,use_cuda=True,device = self.device)
        #这一步超级牛逼了，靠这个Model_feature 提取特征做成特征矩阵

        """在CPU上使用ESM提取氨基酸残基级特征"""
        # import esm
        # data = [("proteinseq",seq)]
        # model_location = "esm2_t6_8M_UR50D"  # 有一个用来平替的大一点的trans模型 esm1b_t33_650M_UR50S.pt
        # model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        # batch_converter = alphabet.get_batch_converter()
        # model.eval()  # disables dropout for deterministic results
        # batch_labels, batch_strs, batch_tokens = batch_converter(data)
        #
        # if (model_location == "esm1b_t33_650M_UR50S"):
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        #     token_representations = results["representations"][33]
        # else:  # 只有6层的esm2_t6_8M_UR50D
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        #     token_representations = results["representations"][6]

        """在GPU上使用ESM提取残基级特征"""
        # import esm
        data = [("proteinseq", seq)]
        device = self.device
        model_location = "esm2_t6_8M_UR50D"  # 有一个大一点的trans模型esm-1b 即esm1b_t33_650M_UR50S.pt，或者用小一点的esm2 即esm2_t6_8M_UR50D.pt
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        model = model.to(device)
        model.eval()

        # 接下来转换输入数据为模型可接受的格式，并移动到 GPU 上
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        if (model_location == "esm1b_t33_650M_UR50S"):
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        else:  # 只有6层的esm2_t6_8M_UR50D
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

        # 将特征移回 CPU（如果需要的话）
        # token_representations = token_representations.cpu()




        # for i, (_, seq) in enumerate(data):
        #     """
        #     这段代码是一个简单的 枚举（enumerate） 循环，用于迭代一个包含蛋白质数据的列表 data。
        #     列表中每个元素是一个元组，包含两个值：第一个值是蛋白质的名称（字符串），第二个值是对应的蛋白质序列（字符串）。
        #
        #     在循环中，enumerate() 函数用于同时获取列表中的索引和对应的元素。
        #     对于每个元素，i 是当前元素的索引，(_, seq) 表示解包元组，忽略第一个值（蛋白质名称），而将第二个值（蛋白质序列）赋值给 seq。
        #     在循环的每次迭代中，你可以使用 seq 变量来访问当前蛋白质序列的值，进行进一步处理或分析。
        #     """
        #
        #     print("index i =", i, "seq=", seq)
        #     print("哈哈我要fa癫了", token_representations[i, 1: len(seq) + 1].shape)
        #     feature_matrix = token_representations[i, 1: len(seq) + 1]

        feature_matrix = token_representations[0, 1: len(seq) + 1]

        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])


        #embedding = torch.Tensor([embedding])


        return GCNData_mol




class ESM_bilstm_Dataset(InMemoryDataset): # 定义子类ESMDataset继承InMemoryDataset
    def __init__(self,model_feature=None,train_data_X=None,train_contactmap_X=None,train_feature_matrix_X=None,train_data_Y=None,root='/tmp',transform=None,pre_transform=None,device = None):
        super(ESM_bilstm_Dataset,self).__init__(root,transform,pre_transform)
        #self.dir=dir
        self.device = device
        self.model_feature = model_feature
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y
        self.X_contactmap_list = train_contactmap_X
        self.X_feature_matrix_list = train_feature_matrix_X
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def __len__(self):
        return int(len(self.X_data_list))

    def __getitem__(self, idx):
        """
        在 Python 中，双下划线 __ 用于标识特殊方法或属性，这些方法和属性在类中有特殊的含义和用途。
        __getitem__ 是 Python 中的一个特殊方法，用于实现索引访问操作，即通过类似 obj[key] 的方式来获取对象的元素。

        根据给定索引 idx 返回数据集中的一个样本。它首先从传入的索引 idx 中获取接触图、标签和特征矩阵，然后将特征矩阵转换为特征表示，
        最后将所有这些数据打包成一个 GCNData_mol 对象返回，其实就是将初始数据做成geometric的输入格式DATA.Data:
        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])
        """

        contact_map = self.X_contactmap_list[self.X_data_list[idx]]
        label = self.Y_data_list[self.X_data_list[idx]]
        seq = self.X_feature_matrix_list[self.X_data_list[idx]] #索引到的就是train_feature_matrix_X 即该条蛋白质的氨基酸序列

        seq_bytes = bytes(seq,encoding='utf-8')
        feature_matrix1 = embed_sequence(self.model_feature, seq_bytes, use_cuda=True, device=self.device) ##这一步很关键，靠这个通过BiLSTM提取特征做成特征矩阵1，L*6165(L为蛋白质氨基酸残基序列的长度，6165为BiLSTM提取的每个残基的特征维度)


        #print("seq=",seq) # 看看这个seq到底是啥



        """在CPU上使用ESM提取氨基酸残基级特征"""
        # import esm
        # data = [("proteinseq",seq)]
        # model_location = "esm2_t6_8M_UR50D"  # 有一个用来平替的大一点的trans模型 esm1b_t33_650M_UR50S.pt
        # model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        # batch_converter = alphabet.get_batch_converter()
        # model.eval()  # disables dropout for deterministic results
        # batch_labels, batch_strs, batch_tokens = batch_converter(data)
        #
        # if (model_location == "esm1b_t33_650M_UR50S"):
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        #     token_representations = results["representations"][33]
        # else:  # 只有6层的esm2_t6_8M_UR50D
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        #     token_representations = results["representations"][6]

        """在GPU上使用ESM提取残基级特征"""
        # import esm
        data = [("proteinseq", seq)]
        device = self.device
        model_location = "esm2_t6_8M_UR50D"  # 有一个的大一点的trans模型 esm1b_t33_650M_UR50S.pt，或者小一点的用来平替的esm2 即esm2_t6_8M_UR50D
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        model = model.to(device)
        model.eval()

        # 接下来转换输入数据为模型可接受的格式，并移动到 GPU 上
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        if (model_location == "esm1b_t33_650M_UR50S"):
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        else:  # 只有6层的esm2_t6_8M_UR50D
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

        # 将特征移回 CPU（如果需要的话）
        # token_representations = token_representations.cpu()




        # for i, (_, seq) in enumerate(data):
        #     """
        #     这段代码是一个简单的 枚举（enumerate） 循环，用于迭代一个包含蛋白质数据的列表 data。
        #     列表中每个元素是一个元组，包含两个值：第一个值是蛋白质的名称（字符串），第二个值是对应的蛋白质序列（字符串）。
        #
        #     在循环中，enumerate() 函数用于同时获取列表中的索引和对应的元素。
        #     对于每个元素，i 是当前元素的索引，(_, seq) 表示解包元组，忽略第一个值（蛋白质名称），而将第二个值（蛋白质序列）赋值给 seq。
        #     在循环的每次迭代中，你可以使用 seq 变量来访问当前蛋白质序列的值，进行进一步处理或分析。
        #     """
        #
        #     print("index i =", i, "seq=", seq)
        #     print("哈哈我要fa癫了", token_representations[i, 1: len(seq) + 1].shape)
        #     feature_matrix = token_representations[i, 1: len(seq) + 1]

        feature_matrix2 = token_representations[0, 1: len(seq) + 1] #这是由ESM2提取出来的残基级别特征矩阵2，L*320(L为蛋白质氨基酸残基序列的长度，320为ESM2提取的每个残基的特征)
        combined_feature_matrix = torch.cat((feature_matrix1, feature_matrix2), dim=1) # 按列拼接特征张量，组成混合特征矩阵 L*(6165+320) 前6165列来自于Bilstm特征提取器，后320列来自于ESM2特征提取器
        """
        在PyTorch中，dim参数用于指定在哪个维度上进行张量的连接。在这个情况下，dim=1表示在张量的第二个维度上进行连接，即在列方向上进行连接。因为Python中索引是从0开始的，所以第一个维度是0，第二个维度是1，以此类推。
        """

        GCNData_mol = DATA.Data(x=combined_feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]),uid = self.X_data_list[idx])


        #embedding = torch.Tensor([embedding])


        return GCNData_mol




def collate(data_list):
    mol_data_list = data_list
    batchA = Batch.from_data_list(mol_data_list)
    #embedding = [data[1] for data in data_list]
    #embedding = torch.stack(embedding).squeeze(dim=1)

    return batchA
#################################################################


def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max

def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(str(num_pos))
    perf["M-aupr"] /= n
    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha * [1])


    perf['F-max'] = calculate_fmax(y_score, y_test)

    return perf



def get_results(Y_test, y_score):
    #print(Y_test.shape)
    #print(y_score.shape)
    perf = defaultdict(dict)
    perf['all'] = evaluate_performance(Y_test, y_score)

    return perf


