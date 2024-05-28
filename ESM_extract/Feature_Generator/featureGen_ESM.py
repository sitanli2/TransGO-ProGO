import numpy as np
import torch, os
import torch.nn as nn
# from utils import get_project_root
import esm
from Esm_for_feature import L1, L2, OrdinalRegression, BilinearContactMap
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """
        encode a byte string into alphabet indices
        即将字节字符串编码为字母表索引，你可以将字节转换为字符，然后将每个字符映射到字母表中的相应索引。
        """

        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)


class Uniprot21(Alphabet):
    def __init__(self, mask=False):
        chars = alphabet = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(chars, encoding=encoding, mask=mask, missing=20)


def embed_sequence(model, x, pool='none', use_cuda=False, device=None):
    # device = torch.device('cuda:1')
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1, n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.to(device)

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z, _ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        # z = z.cpu().numpy()

    return z


class ProSEMT(nn.Module):
    def __init__(self, embedding, scop_predict, cmap_predict):
        super(ProSEMT, self).__init__()
        self.embedding = embedding  # 通过debug可以发现 embedding 其实就是下面的encoder
        self.scop_predict = scop_predict
        self.cmap_predict = cmap_predict



    @staticmethod
    def load_pretrained_ESM():
        # Load ESM-1b model from local
        model_location = "esm1b_t33_650M_UR50S"
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results



    def load_pretrained(path='prose_mt'):
        """加载预训练好的蛋白质语言模型bi-lstm"""
        if path is None or path == 'prose_mt':
            root = get_project_root()
            path = os.path.join(root, 'AlphaFold_refine/saved_models', 'prose_mt_3x1024.sav')

        from Esm_for_feature import SkipLSTM

        encoder = SkipLSTM(21, 21, 1024, 3)  # 实例化SkipLSTM类 nin=nout=21,hidden_dim=1024,num_layers=3
        encoder.cloze = encoder.proj  # 在SkipLSTM的最后一层 self.proj = nn.Linear(n, nout)，最后debug结果显示这个全连接层encoder.proj 输入为6165，输出为21
        # 而恰恰好， The dimension of the residue descriptor obtained by the Bi-LSTM language model is 6165,

        proj_in = encoder.proj.in_features  # 6165
        proj = nn.Linear(proj_in,
                         100)  # 经过debug后发现，proj这个全连接层输入即proj_in = encoder.proj.in_features = 6165，输出正如该行所示被设定为100
        encoder.proj = proj  # 更新encoder即实例化SkipLSTM的proj即全连接层
        encoder.nout = 100

        scop_predict = OrdinalRegression(100, 5, compare=L1(),
                                         allow_insertions=False)  # 正如上面所示输出100再进行OrdinalRegression
        cmap_predict = BilinearContactMap(proj_in)  # proj_in = encoder.proj.in_features = 6165
        model = ProSEMT(encoder, scop_predict, cmap_predict)  # encoder其实就是ProSEMT的embedding（debug 看具体里面是什么）

        state_dict = torch.load(path, map_location=torch.device('cpu'))  # 加载指定路径下的预训练模型
        model.load_state_dict(state_dict)
        """
        这两行代码用于加载预训练模型的参数（state_dict）到当前定义的模型model = ProSEMT(encoder, scop_predict, cmap_predict)中。

        torch.load(path, map_location=torch.device('cpu'))：
        torch.load() 函数用于从指定路径 path 加载保存的对象，其中包括模型的参数等。
        map_location=torch.device('cpu') 的作用是将加载的模型参数映射到 CPU 上，即使之前在 GPU 上训练过也可以正常加载。
        这是因为在加载模型时，如果当前设备不是 GPU，而模型参数保存在 GPU 上，会导致加载失败。所以通过 map_location 参数指定将模型参数加载到 CPU 上。

        model.load_state_dict(state_dict)：
        load_state_dict() 函数用于将加载的参数 state_dict 加载到当前模型 model 中。
        这样，模型就具有了预训练模型的参数。注意，加载的参数必须与当前模型的结构匹配，否则会引发错误。
        具体的load_state_dict() 函数细节得去ctrl进源码里面看
        """

        return model

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def transform(self, x):
        return self.embedding.transform(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)
