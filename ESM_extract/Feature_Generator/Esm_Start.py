import torch
import esm

def esm1bDownload():
    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
    model_location = "esm1b_t33_650M_UR50S"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
    """
    model 是加载后的预训练模型对象。
    这个对象通常是一个神经网络模型，经过预训练以用于特定的生物信息学任务，如蛋白质序列分析、蛋白质结构预测等。
    
    alphabet 是加载后的字母表对象。字母表通常包含了用于表示输入数据的字符集合，例如在蛋白质序列分析中，字母表可以包含氨基酸的缩写。
    这个字母表对象提供了模型需要的字符集合和与之对应的索引映射。
    """
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3",  "K A <mask> I S Q"),
    ]
    """
    atch_tokens：是批量数据的标记化表示。标记化是将原始数据转换为标记（token）序列的过程。
    在文本处理任务中，标记通常是文本中的单词或者子词。因此，batch_tokens 可能是一个二维列表，其中每个元素都是一个标记序列，表示对应的数据样本的标记化表示。
    对蛋白质序列来说，token通常指序列中的氨基酸，因此Debug后发现batch_tokens = {Tensor:{4,73}} 4代表四条序列，4个73维的token表示四条蛋白质序列
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging（均值）
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    import matplotlib.pyplot as plt
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()


# Script to test esm
def test_esm():
    # Load ESM-1b model from local
    model_location = "esm2_t6_8M_UR50D" # 有一个的大一点的trans模型 esm1b_t33_650M_UR50S.pt，或者小一点的用来平替的esm2 即esm2_t6_8M_UR50D
    # model_location = "esm2_t6_8M_UR50D" #有一个用来平替的大一点的模型 esm1b_t33_650M_UR50S.pt
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data_bp (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        # ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        # ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),#H 被换成了<mask>
        # ("protein3", "K A <mask> I S Q"),
    ]

    for i, j in data:
        print(i,"的序列长度=",len(j))

    # seqdata = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    # batch_tokens = batch_converter(seqdata)
    # print("only seq:", batch_tokens)

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    print("batch_labels、strs、tokens",batch_labels,batch_strs,batch_tokens)




    # Extract per-residue representations (on CPU)
    if (model_location == "esm1b_t33_650M_UR50S"):
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
    else: #只有6层的esm2_t6_8M_UR50D
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]

    print('提取出来的per-residue representations 的torch张量的shape：',token_representations.shape)
    """
    最后的输出是：torch.Size([4, 73, 1280]) 这意味着这个张量具有 4 个元素（对应四条氨基酸序列），每个元素都是一个大小为 (73, 1280) 的二维数组
    之后下一步将其做均值化处理，将其映射为一个1028维的向量，作为最后的整条氨基酸序列的特征
    
    同理，只有6层的esm2_t6_8M_UR50D torch.Size([2, 73, 320]) 最后提取的残基级特征为（73*320）的张量
    """


    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        """
        这段代码是一个简单的 枚举（enumerate） 循环，用于迭代一个包含蛋白质数据的列表 data。
        列表中每个元素是一个元组，包含两个值：第一个值是蛋白质的名称（字符串），第二个值是对应的蛋白质序列（字符串）。
        
        在循环中，enumerate() 函数用于同时获取列表中的索引和对应的元素。
        对于每个元素，i 是当前元素的索引，(_, seq) 表示解包元组，忽略第一个值（蛋白质名称），而将第二个值（蛋白质序列）赋值给 seq。
        在循环的每次迭代中，你可以使用 seq 变量来访问当前蛋白质序列的值，进行进一步处理或分析。
        """

        print("index i =",i,"seq=",seq)
        print("哈哈我要癫了",token_representations[i, 1: len(seq) + 1].shape)
        sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))

    #print("通过均值化提取出上述4条序列每条的特征：",sequence_representations,"\n他们的长度分别为：")
    for i in sequence_representations:
        print("ceshi",i.shape)
        print(len(i)) # 都等于1280

    """
    这个1280正好对应ESM-1b 官方文档中给的Embedding Dim = 1280 ，这个1280维的向量就是一条蛋白质氨基酸序列通过ESM提取出来的特征
    """


def esm_extract(seq):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (device == 'cpu'):
        """在CPU上使用ESM提取氨基酸残基级特征"""
        # import esm
        data = [("proteinseq",seq)]
        model_location = "esm2_t6_8M_UR50D"  # 有一个用来平替的大一点的trans模型 esm1b_t33_650M_UR50S.pt
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if (model_location == "esm1b_t33_650M_UR50S"):
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        else:  # 只有6层的esm2_t6_8M_UR50D
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

    else:
        """在GPU上使用ESM提取残基级特征"""
        # import esm
        # seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        data = [("proteinseq", seq)]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_location = "esm1b_t33_650M_UR50S"  # 有一个的大一点的trans模型 esm1b_t33_650M_UR50S.pt，或者小一点的用来平替的esm2 即esm2_t6_8M_UR50D
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
        print('提取出来的per-residue representations 的torch张量的shape：', token_representations.shape)

    feature_matrix = token_representations[0, 1: len(seq) + 1] #是一个 seqlength * 1280的张量，这就是整个蛋白质序列的特征矩阵，每一行代表一个氨基酸残基级特征为1280维。
    print("feature_matrix torch size = ",feature_matrix.shape)
    return feature_matrix

def save_tensor(filepath,filename):
    """保存提取出来的特征张量到本地去"""
    # default file path :'G:/fkfk/AlphaFold_refine/ESM_extract/Feature_Generator/feature_matrix/token_representations.pth'
    # file name = 提取特征的蛋白质序列的UPID 如 P05067.pth
    feature_matrix = esm_extract("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    Intact_filepath = filepath + filename
    torch.save(feature_matrix, Intact_filepath)

def load_tensor(IntactFilepath):
    feature_matrix = torch.load('token_representations.pth')
    return feature_matrix



if __name__ == '__main__':
    # esm1bDownload()
    # test_esm()
    feature_matrix = esm_extract("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    # 先保存张量到本地文件中
    #torch.save(feature_matrix, 'G:/fkfk/AlphaFold_refine/ESM_extract/Feature_Generator/feature_matrix/token_representations.pth')
    # 后续要再想从本地加载张量
    # token_representations = torch.load('token_representations.pth') # 现在你可以将加载的张量作为深度学习网络的输入

