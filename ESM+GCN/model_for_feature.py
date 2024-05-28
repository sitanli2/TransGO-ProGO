import torch.nn as nn
import torch,os
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from pathlib import Path
def get_project_root() -> Path:
    return Path(__file__).parent.parent
def pad_gap_scores(s, gap):
    col = gap.expand(s.size(0), 1)
    s = torch.cat([s, col], 1)
    row = gap.expand(1, s.size(1))
    s = torch.cat([s, row], 0)
    return s
class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1)-y)**2, -1) #PyTorch 中的张量操作，用于计算两组数据点之间的欧氏距离的平方的相反数。
        #综合起来，这个操作的结果是返回一个形状为 (batch_size, num_points) 的张量，其中每个元素表示 x 中每个数据点与 y 中每个数据点之间的欧氏距离的平方。
"""
(x.unsqueeze(1)-y)：这一步是将张量 x 和 y 进行广播（broadcasting），使它们的维度能够匹配。具体来说，它将张量 x 扩展（unsqueeze）一个维度，使其形状变为 (batch_size, 1, features)，其中 batch_size 是数据点的数量，features 是每个数据点的特征数。然后，它从 y 中减去这个扩展后的 x，从而实现了广播操作，使得两个张量的维度能够匹配。

**2：这一步是对每个差值取平方。

torch.sum(..., -1)：这一步是对每个数据点的平方差值沿着最后一个维度（即特征维度）进行求和。这样做的结果是得到了每个数据点与另一组数据点之间的距离的平方。

综合起来，这个操作的结果是一个形状为 (batch_size, num_points) 的张量，其中每个元素表示 x 中每个数据点与 y 中每个数据点之间的欧氏距离的平方的相反数。
"""

class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1) # 这个操作是计算两组数据点之间的曼哈顿距离的相反数。
"""
(x.unsqueeze(1)-y)：这一步同样是将张量 x 和 y 进行广播操作，使它们的维度能够匹配。具体来说，它将张量 x 扩展（unsqueeze）一个维度，使其形状变为 (batch_size, 1, features)，其中 batch_size 是数据点的数量，features 是每个数据点的特征数。然后，它从 y 中减去这个扩展后的 x，实现了广播操作，使得两个张量的维度能够匹配。

torch.abs(...)：这一步是取每个差值的绝对值。

torch.sum(..., -1)：这一步是对每个数据点的绝对差值沿着最后一个维度（即特征维度）进行求和。这样得到的结果是每个数据点与另一组数据点之间的曼哈顿距离。

-torch.sum(...)：最后一步是对上述结果取负数，得到曼哈顿距离的相反数。

综合起来，这个操作的结果是一个形状为 (batch_size, num_points) 的张量，其中每个元素表示 x 中每个数据点与 y 中每个数据点之间的曼哈顿距离的相反数。
"""

class OrdinalRegression(nn.Module):
    def __init__(self, embed_dim, n_classes, compare=L1()
                 , transform=None, align_method='ssa', beta_init=None
                 , allow_insertions=False, gap_init=-10
                 ):
        super(OrdinalRegression, self).__init__()

        self.n_in = embed_dim
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))
        self.transform = transform

        if beta_init is None:
            # set beta to expectation of comparison
            # assuming embeddings are unit normal

            if type(compare) is L1: #L1欧氏距离
                ex = 2 * np.sqrt(2 / np.pi) * embed_dim  # expectation for L1
                var = 4 * (1 - 2 / np.pi) * embed_dim  # variance for L1
            elif type(compare) is L2: #L2曼哈顿距离
                ex = 4 * embed_dim  # expectation for L2
                var = 32 * embed_dim  # variance for L2
            else:
                ex = 0
                var = embed_dim

            beta_init = ex / np.sqrt(var)

        self.theta = nn.Parameter(torch.ones(1, n_classes - 1) / np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) + beta_init)

        self.clip()

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):
        return self.score(z_x, z_y)

    def score(self, z_x, z_y):

        s = self.compare(z_x, z_y)
        if self.allow_insertions:
            s = pad_gap_scores(s, self.gap)

        if self.align_method == 'ssa':
            a = torch.softmax(s, 1)
            b = torch.softmax(s, 0)

            if self.allow_insertions:
                index = s.size(0) - 1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)

                index = s.size(1) - 1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)

            a = a + b - a * b
            a = a / torch.sum(a)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)

        a = a.view(-1, 1)
        s = s.view(-1, 1)

        if hasattr(self, 'transform'):
            if self.transform is not None:
                s = self.transform(s)

        c = torch.sum(a * s)
        logits = c * self.theta + self.beta
        return logits.view(-1)

class BilinearContactMap(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W z_j + b)
    """
    def __init__(self, embed_dim, hidden_dim=1000, width=7, act=nn.LeakyReLU()):
        super(BilinearContactMap, self).__init__()

        self.scale = np.sqrt(hidden_dim)
        self.linear = nn.Linear(embed_dim, embed_dim) #, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def clip(self):
        pass

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        z_flat = z.view(-1, z.size(2))
        h = self.linear(z_flat).view(z.size(0), z.size(1), -1)
        s = torch.bmm(h, z.transpose(1,2))/self.scale + self.bias
        return s
"""
定义了一个名为 BilinearContactMap 的 PyTorch 模型类，用于预测接触图（contact maps）。接触图通常用于描述蛋白质结构中的原子之间的接触关系。
该模型使用双线性函数的形式进行预测，其输出通过 Sigmoid 函数归一化到 [0, 1] 区间。

下面是对代码中各部分的解释：
__init__(self, embed_dim, hidden_dim=1000, width=7, act=nn.LeakyReLU())：
模型的初始化函数，接受输入参数 embed_dim（嵌入维度）、hidden_dim（隐藏层维度，默认为 1000）、width（宽度，默认为 7）和 act（激活函数，默认为 LeakyReLU）。

super(BilinearContactMap, self).__init__()：
调用父类 nn.Module 的初始化方法，初始化神经网络模型。

self.scale = np.sqrt(hidden_dim)：
初始化缩放因子，用于缩放双线性层的输出。

self.linear = nn.Linear(embed_dim, embed_dim)：
定义一个线性层，用于将输入数据进行线性变换。embed_dim 是输入和输出的维度。

self.bias = nn.Parameter(torch.zeros(1))：
定义一个偏置参数，用于模型输出的偏置。

def forward(self, z)：
前向传播函数，接收输入张量 z，并调用 predict 函数进行预测。

def predict(self, z)：
预测函数，接收输入张量 z，将其展平后经过线性层和双线性函数计算，最终返回预测结果。

z_flat = z.view(-1, z.size(2))：
将输入张量 z 进行展平，以便后续的线性变换。

h = self.linear(z_flat).view(z.size(0), z.size(1), -1)：
通过线性层进行线性变换，并将结果重新调整为原始形状。

s = torch.bmm(h, z.transpose(1,2))/self.scale + self.bias：
计算双线性函数的输出，其中 torch.bmm 是批量矩阵乘法，z.transpose(1,2) 将输入张量进行转置，self.scale 是缩放因子，self.bias 是偏置参数。

这个模型主要用于处理蛋白质结构中的数据，并根据输入数据预测原子之间的接触关系，是一种常见的生物信息学模型。

"""




class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin #21
        for i in range(num_layers): #一个循环，用于创建多个 LSTM 层
            f = nn.LSTM(dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional) #这个要去看一下LSTM的复现代码了
            """
            创建一个 LSTM 层。参数含义如下：
            dim 是输入数据的特征维度（input size），在第一次循环中，它是输入数据的维度，而后续循环中它会根据是否是双向 LSTM 而改变。
            hidden_dim 是 LSTM 隐藏状态的维度（hidden size），它决定了 LSTM 内部状态的大小。
            1 表示 LSTM 层的层数。
            batch_first=True 表示输入数据的形状是 (batch_size, seq_length, dim)，即批次维度在第一维度。
            bidirectional=bidirectional 表示是否使用双向 LSTM。如果设置为 True，则创建的 LSTM 是双向的，否则是单向的。
            
            因为：hidden_dim=1024，bidirectional=True，并且有 num_layers=3 层 LSTM，那么该网络的结构如下：

            第一个 LSTM 层的输入特征维度是 21，隐藏状态的维度是 1024（因为dim=nin=21；hidden_dim=1024），并且是双向的。
            第二个 LSTM 层的输入特征维度是 2048（双向 LSTM 的隐藏状态维度的两倍），隐藏状态的维度仍然是 1024，因为在这个例子中不改变隐藏状态维度。
            第三个 LSTM 层的输入特征维度同样是 2048，隐藏状态的维度仍然是 1024。
            这个网络的每一层都会有一个隐藏状态的维度为 1024，每个时间步的输出维度也是 1024（因为是双向 LSTM，所以输出的维度是隐藏状态的两倍）。
            """
            self.layers.append(f)
            if bidirectional:
                dim = 2*hidden_dim #2*1024=2048
            else:
                dim = hidden_dim

        n = hidden_dim*num_layers + nin
        if bidirectional:
            n = 2*hidden_dim*num_layers + nin # n= 2*1024*3+21 = 6165!!!!!!!!!!!!!!!!!

        self.proj = nn.Linear(n, nout) #全连接层？又通过上面的代码可知 n= 2*1024*3+21 = 6165!!!!!!!!!!!!!!!!!

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        if path is None or path == 'prose_dlm':
            root = get_project_root()
            path = os.path.join(root, 'saved_models', 'prose_dlm_3x1024.sav')

        model = SkipLSTM(21, 21, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs =  [one_hot] # []
        h_ = one_hot
        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1,h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z
