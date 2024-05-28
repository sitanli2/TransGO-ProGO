import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool as gep,global_max_pool as gmp,global_add_pool as gap
from torch_geometric.nn import GCNConv  # 直接调包，所以看懂torch-geometric的GCN解释文档尤其重要：https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#
from torch_geometric.nn import SAGEConv # GCNConv的平替（谁更好需要进一步验证）
from torch_geometric.nn import GATv2Conv # fixes the static attention problem of the standard GATConv layer.


class Model_Net(torch.nn.Module):
    def __init__(self, n_output=None,num_features=6165, output_dim=486, dropout=0.2,aa_num=21,hidden_dim = 128,net = None,pool=None): #未精炼前的数据有486条mf标签 output_dim=486
        super(Model_Net, self).__init__()
        self.pool = pool
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1024)
        self.n_output = n_output
        #mol network
        '''
        由于HLM的前21维( 20种标准氨基酸和1种未知氨基酸)是序列的one - hot表示，所以我们要对它们单独进行处理
        aa_num=21
        torch.nn.Linear 用于创建线性变换（也称为全连接层或仿射层）
        '''
        self.feature_linear1 = torch.nn.Linear(num_features-aa_num,hidden_dim) #在这个实验中，该全连接层输入为6144，输出为512
        self.feature_linear2 = torch.nn.Linear(aa_num, aa_num) #该全连接层输入为21，输出也为21
        # 上面两个全连接层的输出再做并置操作，512+21 = 533 即为后面GCN第一层的输入
        if self.net == 'GCN':
            self.prot_conv1 = GCNConv(hidden_dim+aa_num, hidden_dim+aa_num)
            self.prot_conv2 = GCNConv(hidden_dim+aa_num, (hidden_dim+aa_num) * 2)
            self.prot_conv3 = GCNConv((hidden_dim+aa_num) * 2, (hidden_dim+aa_num) * 4)
        elif self.net == 'GAT':
            self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
            '''
            GAT 是图注意力网络（Graph Attention Network）的缩写。GAT 是一种深度学习模型，用于处理图结构数据。它是一种图神经网络（Graph Neural Network，GNN）的变体，旨在学习节点之间的关系和特征表示。

            GAT 最初由Petar Velickovic等人在2017年的论文《Graph Attention Networks》中提出。它引入了一种称为"自注意力"机制的方法，允许每个节点根据其邻居节点的特征来动态地分配注意力权重。这种自注意力机制使得模型能够在处理图数据时更加灵活和有效地捕捉节点之间的关系，从而提高了模型的性能。
            
            与传统的图卷积网络（GCN）相比，GAT 具有以下一些优势：
            
            灵活的邻居关系建模：GAT 使用注意力机制来动态地调整节点之间的关系权重，因此可以更好地适应不同节点之间的关系特征。
            
            并行计算：由于注意力权重是针对每个节点与其邻居节点之间计算的，因此可以并行计算，从而提高了训练和推断的效率。
            
            节点特征的自适应性：GAT 能够对每个节点动态地学习特征权重，因此可以更好地适应不同节点的特征表示。
            
            由于这些优点，GAT 在图数据领域得到了广泛的应用，如社交网络分析、推荐系统、生物信息学等。
            '''

        """第一个全连接层"""
        if '-' in self.pool:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim + aa_num) * 4 * 2, 1024)
        else:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim+aa_num) * 4, 1024)
        """第二个全连接层"""
        self.prot_fc_g2 = torch.nn.Linear(1024,  output_dim) #实验中这最后一个全连接层的output_dim = 486 即对应 mf标签的总数，好最后通过sigmoid对每个标签输出他们的预测概率



    def forward(self, data_prot):
        prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
        prot_feature1 = self.relu(self.feature_linear1(prot_x[:,21:]))
        prot_feature2 = self.relu(self.feature_linear2(prot_x[:,:21]))
        prot_feature = torch.cat((prot_feature2,prot_feature1),1)
        x = self.prot_conv1(prot_feature,prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv2(x,prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv3(x,prot_edge_index)
        x = self.relu(x)
        if self.pool == 'gep':
            x = gep(x, prot_batch)
        elif self.pool == 'gmp':
            x = gmp(x, prot_batch)
        elif self.pool == 'gap':
            x = gap(x, prot_batch)
        elif self.pool == 'gap-gmp':
            x1 = gap(x, prot_batch)
            x2 = gmp(x, prot_batch)
            x = torch.cat((x1,x2),1)
        elif self.pool == 'gap-gep':
            x1 = gap(x, prot_batch)
            x2 = gep(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        x = self.prot_fc_g1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.prot_fc_g2(x)
        x = self.sigmoid(x)

        return x


        #return out





"""接下来是适用与ESM输入特征的GCN模型
此时不再需要和之前预训练的Bi-LSTM模型一样考虑 前21个氨基酸残基了，直接给他做一个线性变化降维就行，再通过两个全连接层
"""
class Model_Net_ESM(torch.nn.Module):
    def __init__(self, n_output=None, num_features=320, output_dim=486, dropout=0.2, aa_num=21, hidden_dim=128,net=None, pool=None):  # 未精炼前的数据有486条mf标签 output_dim=486，精炼之后则是482
        super(Model_Net_ESM, self).__init__()
        self.pool = pool
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1024)
        self.n_output = n_output
        # mol network
        '''
        由于HLM的前21维( 20种标准氨基酸和1种未知氨基酸)是序列的one - hot表示，所以我们要对它们单独进行处理
        aa_num=21
        torch.nn.Linear 用于创建线性变换（也称为全连接层或仿射层）
        '''
        # self.feature_linear1 = torch.nn.Linear(num_features - aa_num, hidden_dim)  # 在这个实验中，该全连接层输入为6144，输出为512
        # self.feature_linear2 = torch.nn.Linear(aa_num, aa_num)  # 该全连接层输入为21，输出也为21
        # # 上面两个全连接层的输出再做并置操作，512+21 = 533 即为后面GCN第一层的输入

        self.feature_linear = torch.nn.Linear(num_features,hidden_dim) #ESM2输入的特征为320维，用这个线性变换层给他降维，同理ESM-1b的输入特征为1280，也可以这样帮他降维，降到hidden_dim个维度
        if self.net == 'GCN':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        elif self.net == 'GAT':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATConv(hidden_dim * 2,hidden_dim * 4)

            # self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            # self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            # self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
        elif self.net == 'GCN_GCNConv+GAT_GATv2Conv':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)
        elif self.net == 'GCN_SAGEConv+GAT_GATv2Conv':
            self.prot_conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.prot_conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)


        """第一个全连接层"""
        if '-' in self.pool:
            self.prot_fc_g1 = torch.nn.Linear(hidden_dim * 4 * 2, 1024)
        else:
            self.prot_fc_g1 = torch.nn.Linear(hidden_dim * 4, 1024)
        """第二个全连接层"""
        self.prot_fc_g2 = torch.nn.Linear(1024,output_dim)  # 实验中这最后一个全连接层的output_dim = 486 即对应 mf标签的总数，好最后通过sigmoid对每个标签输出他们的预测概率

    def forward(self, data_prot):
        prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
        # prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:]))
        # prot_feature2 = self.relu(self.feature_linear2(prot_x[:, :21]))
        # prot_feature = torch.cat((prot_feature2, prot_feature1), 1)
        prot_feature = self.relu(self.feature_linear(prot_x)) #直接线性变换，不需要考虑前21维是氨基酸残基
        x = self.prot_conv1(prot_feature, prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv2(x, prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv3(x, prot_edge_index)
        x = self.relu(x)
        if self.pool == 'gep':
            x = gep(x, prot_batch)
        elif self.pool == 'gmp':
            x = gmp(x, prot_batch)
        elif self.pool == 'gap':
            x = gap(x, prot_batch)
        elif self.pool == 'gap-gmp':
            x1 = gap(x, prot_batch)
            x2 = gmp(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        elif self.pool == 'gap-gep':
            x1 = gap(x, prot_batch)
            x2 = gep(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        x = self.prot_fc_g1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.prot_fc_g2(x)
        x = self.sigmoid(x)

        return x

        # return out



"""
接下来是使用ESM+biLSTM混合特征作为输入的GATv2及其GCN模型
"""
# class Model_Net_ESM_biLSTM(torch.nn.Module):
#     def __init__(self, n_output=None, num_features=6165+320, output_dim=486, dropout=0.2, aa_num=21, hidden_dim=128,net=None, pool=None):  # 未精炼前的数据有486条mf标签 output_dim=486，精炼之后则是482
#         # 此时的输入特征是混合特征
#         super(Model_Net_ESM_biLSTM, self).__init__()
#         self.pool = pool
#         self.net = net
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.sigmoid = nn.Sigmoid()
#         self.bn = nn.BatchNorm1d(1024)
#         self.n_output = n_output
#         # mol network
#         '''
#         由于bilstm提取的特征的前21维( 20种标准氨基酸和1种未知氨基酸)是序列的one - hot表示，所以我们要对它们单独进行处理
#         aa_num=21
#         torch.nn.Linear 用于创建线性变换（也称为全连接层或仿射层）
#         '''
#         self.feature_linear1 = torch.nn.Linear(num_features - aa_num, hidden_dim)  # 在这个实验中，该全连接层输入为6144+320，输出为512
#         self.feature_linear2 = torch.nn.Linear(aa_num, aa_num)  # 该全连接层输入为21，输出也为21
#         # self.feature_linear3 = torch.nn.Linear(num_features, hidden_dim)  # ESM2输入的特征为320/1280维，用这个线性变换层给他降维，降到hidden_dim个维度，这样后面的GCN输入就是2*hidden_dim + aa_num
#         # # 上面两个全连接层的输出再做并置操作，512+21 = 533 即为后面GCN第一层的输入
#
#         # self.feature_linear = torch.nn.Linear(num_features,hidden_dim) #ESM2输入的特征为320维，用这个线性变换层给他降维，同理ESM-1b的输入特征为1280，也可以这样帮他降维，降到hidden_dim个维度
#         if self.net == 'GCN':
#             self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
#             self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
#             self.prot_conv3 = GCNConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
#         elif self.net == 'GAT':
#             self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
#             self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
#             self.prot_conv3 = GATConv(hidden_dim * 2,hidden_dim * 4)
#
#             # self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
#             # self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
#             # self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
#         elif self.net == 'GCN_GCNConv+GAT_GATv2Conv':
#             self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
#             self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
#             self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)
#         elif self.net == 'GCN_SAGEConv+GAT_GATv2Conv':
#             self.prot_conv1 = SAGEConv(hidden_dim, hidden_dim)
#             self.prot_conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
#             self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)
#
#
#         """第一个全连接层"""
#         if '-' in self.pool:
#             self.prot_fc_g1 = torch.nn.Linear((hidden_dim + aa_num) * 4 * 2, 1024)
#         else:
#             self.prot_fc_g1 = torch.nn.Linear((hidden_dim+aa_num) * 4, 1024)
#         """第二个全连接层"""
#         self.prot_fc_g2 = torch.nn.Linear(1024,output_dim)  # 实验中这最后一个全连接层的output_dim = 486 即对应 mf标签的总数，好最后通过sigmoid对每个标签输出他们的预测概率
#
#     def forward(self, data_prot):
#         prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
#         prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:])) #前21维为BiLSTM网络提取的
#         prot_feature2 = self.relu(self.feature_linear2(prot_x[:, :21]))
#         # prot_feature2 = self.relu(self.feature_linear2(prot_x[:, 21:6165])) #22-6165 为bilstm提取特征的非one-hot编码部分
#         # prot_feature3 = self.relu(self.feature_linear2(prot_x[:, 6165:6165+320])) #后面的320/1280维是由ESM2提取的特征，我们也直接对他做一个可学习的线性变换dense且降维
#         # prot_feature3 = self.relu(self.feature_linear3(prot_x[:, 21:6165])))
#         prot_feature = torch.cat((prot_feature2, prot_feature1), 1)
#         # prot_feature = torch.cat((prot_feature2, prot_feature1,prot_feature3), 1)
#
#
#         x = self.prot_conv1(prot_feature, prot_edge_index)
#         x = self.relu(x)
#
#         x = self.prot_conv2(x, prot_edge_index)
#         x = self.relu(x)
#
#         x = self.prot_conv3(x, prot_edge_index)
#         x = self.relu(x)
#         if self.pool == 'gep':
#             x = gep(x, prot_batch)
#         elif self.pool == 'gmp':
#             x = gmp(x, prot_batch)
#         elif self.pool == 'gap':
#             x = gap(x, prot_batch)
#         elif self.pool == 'gap-gmp':
#             x1 = gap(x, prot_batch)
#             x2 = gmp(x, prot_batch)
#             x = torch.cat((x1, x2), 1)
#         elif self.pool == 'gap-gep':
#             x1 = gap(x, prot_batch)
#             x2 = gep(x, prot_batch)
#             x = torch.cat((x1, x2), 1)
#         x = self.prot_fc_g1(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.prot_fc_g2(x)
#         x = self.sigmoid(x)
#
#         return x
#
#         # return out


class Model_Net_ESM_biLSTM_upgrade(torch.nn.Module):
    def __init__(self, n_output=None, num_features=6165+320, output_dim=486, dropout=0.2, aa_num=21, hidden_dim=128,net=None, pool=None):  # 未精炼前的数据有486条mf标签 output_dim=486，精炼之后则是482
        # 此时的输入特征是混合特征
        super(Model_Net_ESM_biLSTM_upgrade, self).__init__()
        self.pool = pool
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1024)
        self.n_output = n_output
        # mol network
        '''
        由于bilstm提取的特征的前21维( 20种标准氨基酸和1种未知氨基酸)是序列的one - hot表示，所以我们要对它们单独进行处理
        aa_num=21
        torch.nn.Linear 用于创建线性变换（也称为全连接层或仿射层）
        '''
        self.feature_linear1 = torch.nn.Linear(num_features - aa_num - 320, hidden_dim)  # 在这个实验中，该全连接层输入为（6144+21+320）这里面的6144，输出为512
        self.feature_linear2 = torch.nn.Linear(aa_num, aa_num)  # 该全连接层输入为21，输出也为21
        # self.feature_linear3 = torch.nn.Linear(320, 320)  # ESM2输入的特征为320/1280维，用这个线性变换层给他降维，通过全连接映射到320个维度，这样后面的GCN输入就是2*hidden_dim + aa_num
        # # 上面两个全连接层的输出再做并置操作，512+21+320 = 853 即为后面GCN第一层混合特征的输入

        # self.feature_linear = torch.nn.Linear(num_features,hidden_dim) #ESM2输入的特征为320维，用这个线性变换层给他降维，同理ESM-1b的输入特征为1280，也可以这样帮他降维，降到hidden_dim个维度
        if self.net == 'GCN':
            self.prot_conv1 = GCNConv(hidden_dim + aa_num + 320, hidden_dim + aa_num + 320)
            self.prot_conv2 = GCNConv(hidden_dim + aa_num + 320, (hidden_dim + aa_num + 320) * 2)
            self.prot_conv3 = GCNConv((hidden_dim + aa_num + 320) * 2, (hidden_dim + aa_num + 320) * 4)
        elif self.net == 'GAT':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATConv(hidden_dim * 2,hidden_dim * 4)

            # self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            # self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            # self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
        elif self.net == 'GCN_GCNConv+GAT_GATv2Conv':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)
        elif self.net == 'GCN_SAGEConv+GAT_GATv2Conv':
            self.prot_conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.prot_conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)


        """第一个全连接层"""
        if '-' in self.pool:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim + aa_num + 320) * 4 * 2, 1024)
        else:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim+aa_num + 320) * 4, 1024)
        """第二个全连接层"""
        self.prot_fc_g2 = torch.nn.Linear(1024,output_dim)  # 实验中这最后一个全连接层的output_dim = 486 即对应 mf标签的总数，好最后通过sigmoid对每个标签输出他们的预测概率

    def forward(self, data_prot):
        prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
        # prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:])) #prot_x[:, 21:]代表从prot_x 中选取所有行（由 : 表示），以及从第21列（即索引为21的列，这里使用零索引）到最后一列的所有列。
        prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:6165])) #prot_x[:, 21:6165]代表prot_x 中选取所有行，以及从第21列（索引为21的列）到第6164列（索引为6164的列）
        prot_feature2 = self.relu(self.feature_linear2(prot_x[:, :21])) #prot_x[:, :21] 从 prot_x 中选取所有行（由 : 表示），以及从第0列（即索引为0的列）到第20列（即索引为20的列，注意这里是闭区间）。
        # prot_feature3 = self.relu(self.feature_linear3(prot_x[:, 6165:])) #prot_x[:, 6165:] 选取所有行，以及从第6165列（索引为6165的列）到最后一列的所有列（即ESM提取的后320列）。
        prot_feature3 = prot_x[:, 6165:]
        # prot_feature2 = self.relu(self.feature_linear2(prot_x[:, 21:6165])) #22-6165 为bilstm提取特征的非one-hot编码部分
        # prot_feature3 = self.relu(self.feature_linear2(prot_x[:, 6165:6165+320])) #后面的320/1280维是由ESM2提取的特征，我们也直接对他做一个可学习的线性变换dense且降维
        # prot_feature3 = self.relu(self.feature_linear3(prot_x[:, 21:6165])))
        prot_feature = torch.cat((prot_feature2, prot_feature1,prot_feature3), 1) #混合特征做上面的可学习非线性映射后再做拼接，作为GCN的真正输入hidden_dim + aa_num + 320 = 512+21+320 = 853
        # prot_feature = torch.cat((prot_feature2, prot_feature1,prot_feature3), 1)


        x = self.prot_conv1(prot_feature, prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv2(x, prot_edge_index)
        x = self.relu(x)

        x = self.prot_conv3(x, prot_edge_index)
        x = self.relu(x)
        if self.pool == 'gep':
            x = gep(x, prot_batch)
        elif self.pool == 'gmp':
            x = gmp(x, prot_batch)
        elif self.pool == 'gap':
            x = gap(x, prot_batch)
        elif self.pool == 'gap-gmp':
            x1 = gap(x, prot_batch)
            x2 = gmp(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        elif self.pool == 'gap-gep':
            x1 = gap(x, prot_batch)
            x2 = gep(x, prot_batch)
            x = torch.cat((x1, x2), 1)
        x = self.prot_fc_g1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.prot_fc_g2(x)
        x = self.sigmoid(x)

        return x

        # return out


"""
接下来我将思考一种新的，结构更为复杂且引入组合图卷积的模型TransGo++
使用ESM+biLSTM混合特征作为特征矩阵输入，使用PDB_contactmap & AlphaFold_contactmap作为邻接矩阵输入 引入由GATv2及其GCN混合构建的组合图卷积网络模型
"""
class Model_Net_ESM_biLSTM_CombineUpgrade(torch.nn.Module):
    def __init__(self, n_output=None, num_features=6165+320, output_dim=486, dropout=0.2, aa_num=21, hidden_dim=128,net=None, pool=None):  # 未精炼前的数据有486条mf标签 output_dim=486，精炼之后则是482
        # 此时的输入特征是混合特征
        super(Model_Net_ESM_biLSTM_CombineUpgrade, self).__init__()
        self.pool = pool
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(1024)
        self.n_output = n_output
        # mol network
        '''
        由于bilstm提取的特征的前21维( 20种标准氨基酸和1种未知氨基酸)是序列的one - hot表示，所以我们要对它们单独进行处理
        aa_num=21
        torch.nn.Linear 用于创建线性变换（也称为全连接层或仿射层）
        '''
        self.feature_linear1 = torch.nn.Linear(num_features - aa_num - 320, hidden_dim)  # 在这个实验中，该全连接层输入为（6144+21+320）这里面的6144，输出为512
        self.feature_linear2 = torch.nn.Linear(aa_num, aa_num)  # 该全连接层输入为21，输出也为21
        self.feature_linear3 = torch.nn.Linear(320, 320)  # ESM2输入的特征为320/1280维，用这个线性变换层给他降维，通过全连接映射到320个维度，这样后面的GCN输入就是2*hidden_dim + aa_num
        # # 上面两个全连接层的输出再做并置操作，512+21+320 = 853 即为后面GCN第一层混合特征的输入

        # self.feature_linear = torch.nn.Linear(num_features,hidden_dim) #ESM2输入的特征为320维，用这个线性变换层给他降维，同理ESM-1b的输入特征为1280，也可以这样帮他降维，降到hidden_dim个维度
        if self.net == 'GCN':
            self.prot_conv1 = GCNConv(hidden_dim + aa_num + 320, hidden_dim + aa_num + 320)
            self.prot_conv2 = GCNConv(hidden_dim + aa_num + 320, (hidden_dim + aa_num + 320) * 2)
            self.prot_conv3 = GCNConv((hidden_dim + aa_num + 320) * 2, (hidden_dim + aa_num + 320) * 4)
        elif self.net == 'GAT':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATConv(hidden_dim * 2,hidden_dim * 4)

            # self.prot_conv1 = GCNConv(hidden_dim + aa_num, hidden_dim + aa_num)
            # self.prot_conv2 = GCNConv(hidden_dim + aa_num, (hidden_dim + aa_num) * 2)
            # self.prot_conv3 = GATConv((hidden_dim + aa_num) * 2, (hidden_dim + aa_num) * 4)
        elif self.net == 'GCN_GCNConv+GAT_GATv2Conv':
            self.prot_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.prot_conv2 = GCNConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)
        elif self.net == 'GCN_SAGEConv+GAT_GATv2Conv':
            self.prot_conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.prot_conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
            self.prot_conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 4)

        elif self.net == 'combine_net':
            self.PDB_prot_conv1 = GCNConv(hidden_dim + aa_num + 320, hidden_dim + aa_num + 320)
            self.PDB_prot_conv2 = GCNConv(hidden_dim + aa_num + 320, (hidden_dim + aa_num + 320) * 2)

            self.AlphaFold_prot_conv1 = GCNConv(hidden_dim + aa_num + 320, hidden_dim + aa_num + 320)
            self.AlphaFold_prot_conv2 = GCNConv(hidden_dim + aa_num + 320, (hidden_dim + aa_num + 320) * 2)

            self.PDB_prot_conv3 = GCNConv((hidden_dim + aa_num + 320) * 4, (hidden_dim + aa_num + 320) * 4) #这一步是将上面两个图卷积的输出组合输入给这个图卷积


        """第一个全连接层"""
        if '-' in self.pool:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim + aa_num + 320) * 4 * 2 , 1024) #组合后，这个也要在之前的((hidden_dim + aa_num + 320) * 4 * 2)基础上
        else:
            self.prot_fc_g1 = torch.nn.Linear((hidden_dim+aa_num + 320) * 4 , 1024) #组合后，这个也要在之前的 (hidden_dim+aa_num + 320) * 4 * 2 基础上
        """第二个全连接层"""
        self.prot_fc_g2 = torch.nn.Linear(1024,output_dim)  # 实验中这最后一个全连接层的output_dim = 486 即对应 mf标签的总数，好最后通过sigmoid对每个标签输出他们的预测概率

    def forward(self, data_prot):
        prot_x, prot_edge_index, prot_batch = data_prot.x, data_prot.edge_index, data_prot.batch
        # prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:])) #prot_x[:, 21:]代表从prot_x 中选取所有行（由 : 表示），以及从第21列（即索引为21的列，这里使用零索引）到最后一列的所有列。
        prot_feature1 = self.relu(self.feature_linear1(prot_x[:, 21:6165])) #prot_x[:, 21:6165]代表prot_x 中选取所有行，以及从第21列（索引为21的列）到第6164列（索引为6164的列）
        prot_feature2 = self.relu(self.feature_linear2(prot_x[:, :21])) #prot_x[:, :21] 从 prot_x 中选取所有行（由 : 表示），以及从第0列（即索引为0的列）到第20列（即索引为20的列，注意这里是闭区间）。
        prot_feature3 = self.relu(self.feature_linear3(prot_x[:, 6165:])) #prot_x[:, 6165:] 选取所有行，以及从第6165列（索引为6165的列）到最后一列的所有列（即ESM提取的后320列）。
        # prot_feature3 = prot_x[:, 6165:] #直接输入ESM提取的特征，不对其做可学习的非线性映射

        # prot_feature2 = self.relu(self.feature_linear2(prot_x[:, 21:6165])) #22-6165 为bilstm提取特征的非one-hot编码部分
        # prot_feature3 = self.relu(self.feature_linear2(prot_x[:, 6165:6165+320])) #后面的320/1280维是由ESM2提取的特征，我们也直接对他做一个可学习的线性变换dense且降维
        # prot_feature3 = self.relu(self.feature_linear3(prot_x[:, 21:6165])))
        prot_feature = torch.cat((prot_feature2, prot_feature1,prot_feature3), 1) #混合特征做上面的可学习非线性映射后再做拼接，作为GCN的真正输入hidden_dim + aa_num + 320 = 512+21+320 = 853
        # prot_feature = torch.cat((prot_feature2, prot_feature1,prot_feature3), 1)


        x = self.PDB_prot_conv1(prot_feature, prot_edge_index)
        x = self.relu(x)

        x = self.PDB_prot_conv2(x, prot_edge_index)
        x = self.relu(x)

        y = self.AlphaFold_prot_conv1(prot_feature,prot_edge_index)
        y = self.relu(y)

        y = self.AlphaFold_prot_conv2(y,prot_edge_index)
        y = self.relu(y)

        GCNconcat = torch.cat((x,y),dim=1)

        z = self.PDB_prot_conv3(GCNconcat, prot_edge_index)
        z = self.relu(z)

        if self.pool == 'gep':
            z = gep(z, prot_batch)
        elif self.pool == 'gmp':
            z = gmp(z, prot_batch)
        elif self.pool == 'gap':
            z = gap(z, prot_batch)
        elif self.pool == 'gap-gmp':
            z1 = gap(z, prot_batch)
            z2 = gmp(z, prot_batch)
            z = torch.cat((z1, z2), 1)
        elif self.pool == 'gap-gep':
            z1 = gap(z, prot_batch)
            z2 = gep(z, prot_batch)
            z = torch.cat((z1, z2), 1)
        z = self.prot_fc_g1(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.prot_fc_g2(z)
        z = self.sigmoid(z)

        return z

        # return out






"""下面是一个应用SAGEConv构建GCN以及用TopKPooling进行对图的下采样(预剪枝)的实例"""
embed_dim = 128
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Net(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)  # ratio=0.8 表示最终想保留80%的点
        '''
        卷积神经网络有pooling ，图卷积也有pooling图中的特征也需要下采样和池化，用的比较多的就是这个
        TopKPooling
        '''
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        '''embedding嵌入'''
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 10, embedding_dim=embed_dim)

        '''接下来全连接做多分类'''
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)  # 全局的平均池化
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        # print('sigmoid',x.shape)
        return x

"""下面是一个简单的使用GCNConv构造GCN网络的实例"""
class GCN_Examp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)  # 只需定义好输入特征和输出特征即可
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)
        '''
        第一层 self.conv1 = GCNConv(dataset.num_features, 4)
        dataset.num_features 就是输入的34维的特征向量，4表示经过这一点后，每个点得到一个4维的向量

        第二层 self.conv2 = GCNConv(4, 4)
        输入为第一层的输出4维，输出也设定为4维

        第三层 self.conv3 = GCNConv(4, 2)
        输入为4，输出为2维，因为待会儿的做一个可视化展示，输出2维向量方便把图画出来

        第四层 self.classifier = Linear(2, dataset.num_classes)
        其实就是全连接层 做分类，2 为输入的2维特征向量 ，dataset.num_classes 则为输出的分类结果
        （一共有4类）的概率值

        '''

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)  # 输入特征与邻接矩阵（注意格式，上面那种）
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        # 分类层
        out = self.classifier(h)

        return out, h

    '''
    forward 顾名思义，就是前向传播
    h = self.conv1(x, edge_index)
    这段代码很重要，上面定义GCN的时候 self.conv1 = GCNConv(dataset.num_features, 4)只定义了
    输入和输出，而GCN的实际输入的是 x 每个点的特征 + edge_index（邻接矩阵，千万不能忘了这个）

    从后续的：
    h = self.conv2(h, edge_index)
    h = self.conv3(h, edge_index)
    也可以看出，无论前向传播时h如何变化，edge_index永远不变，即图（邻接矩阵）的结构无论经过多少层
    GCN层，也永远保持不变。
    x->h->h 每个点的特征都在前向传播的过程中更新

    最后再连一个全连接层输出分类结果
    # 分类层
    out = self.classifier(h)

    最后返回一个最后结果out，以及一个中间结果h()
    return out, h
    h其实就是返回最后分类结果out的前一层 self.conv3 = GCNConv(4, 2)的输出的2维向量，用这个来去
    画图（可视化）

    '''

# if __name__ == '__main__':
#     model = Model_Net()
#     print(model)













