import argparse, os
# from utils import loading_contact_map, loading_seq, get_label, loading_item_data, PFPDataset, collate,ESMDataset,ESM_bilstm_Dataset #大批量测试数据 Database1 or Database1+
from utilsForTest import loading_contact_map, loading_seq, get_label, loading_item_data, PFPDataset, collate,ESMDataset,ESM_bilstm_Dataset #小批量测试数据 testData
from torch.utils.data import DataLoader
from model import Model_Net,Model_Net_ESM,Model_Net_ESM_biLSTM_upgrade,Model_Net_ESM_biLSTM_CombineUpgrade # Model_Net_ESM_biLSTM
from featureGen import ProSEMT
import torch, json
import torch.nn as nn
import torch.optim as optim
from utils import get_results
import warnings
import time


def train(args):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)
    '''
    device = torch.device('cuda:0') 即选用系统上的第一个GPU设备，
    device = torch.device('cuda:1') 即选用系统上的第二个GPU设备。
    '''
    model_feature = ProSEMT.load_pretrained().to(device).eval()

    for data_source in args.data:
        if data_source == 'pdb':
            print('loading contact maps for pdb structures...')
            dict_contact_pdb = loading_contact_map(data_source, args.threshold)
            print('loading seq for pdb structures...')
            dict_seq_pdb = loading_seq(data_source)
        elif data_source == 'alpha':
            print('loading contact maps for alphafold structures...')
            dict_contact_alpha = loading_contact_map(data_source, args.threshold)
            print('loading seq for alphafold structures...')
            dict_seq_alpha = loading_seq(data_source)

    for func in args.func:
        label, label_num = get_label(func)
        train_item = loading_item_data(func, 'train')
        valid_item = loading_item_data(func, 'valid')
        test_item = loading_item_data(func, 'test')
        for data_source in args.data:
            print('############################################################')
            print('training for ' + str(func) + ' using ' + args.net + ',' + str(data_source) + ' data...')
            m_AUPR = -1.0  # AUOR初始值设置为 -1.0
            if data_source == 'pdb':
                train_data = ESM_bilstm_Dataset(train_data_X=train_item, train_contactmap_X=dict_contact_pdb,
                                        train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature, device=device)

                valid_data = ESM_bilstm_Dataset(train_data_X=valid_item, train_contactmap_X=dict_contact_pdb,
                                        train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature, device=device)

                # test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

            elif data_source == 'alpha':
                train_data = PFPDataset(train_data_X=train_item, train_contactmap_X=dict_contact_alpha,
                                        train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature, device=device)

                valid_data = PFPDataset(train_data_X=valid_item, train_contactmap_X=dict_contact_alpha,
                                        train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature, device=device)

                # test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

            dataset_train = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate, drop_last=False)
            dataset_valid = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            """batch_size = 64且使用Esm2来提取特征的话 对我的这台主机CPU来说太大了,改成32甚至16一个batch"""
            # dataset_train = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
            # dataset_valid = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

            # dataset_test = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            """
            这两行代码创建了一个用于训练的数据加载器（DataLoader）：

            train_data 是你的训练数据集。
            batch_size=64 指定了每个批次（batch）中包含的样本数量，这里是 64 个样本。
            shuffle=True 表示在每个 epoch 开始时是否对数据进行洗牌，以增加训练的随机性。将其设置为 True 可以帮助模型更好地泛化，因为它会在每个 epoch 中以不同的顺序遍历数据。
            collate_fn=collate 是一个用于自定义数据加载逻辑的函数，这里的 collate 是一个自定义的函数在（utils中）：

            drop_last=False 指定了当数据集的大小不能被 batch_size 整除时，是否丢弃最后一个不完整的批次。这里设置为 False 表示保留最后一个不完整的批次。
            综合起来，这行代码创建了一个数据加载器 dataset_train，用于从训练数据集中按照指定的 batch_size 加载数据，并在每个 epoch 开始时对数据进行洗牌，同时保留最后一个不完整的批次。
            """

            """
            接下来，定义了一个神经网络GCN模型 model，该模型是通过 Model_Net 类创建的，
            传递了输出维度 label_num、网络结构 args.net、隐藏层维度 args.hidden_dim、池化方式 args.pool 和 dropout 概率 args.dropout 等参数，
            并将模型移动到指定的设备 device 上。
            """
            # model = Model_Net(output_dim=label_num, net=args.net, hidden_dim=args.hidden_dim, pool=args.pool,dropout=args.dropout).to(device)
            model = Model_Net_ESM_biLSTM_CombineUpgrade(output_dim=label_num, net=args.net, hidden_dim=args.hidden_dim, pool=args.pool,dropout=args.dropout).to(device)

            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate)  # 创建了一个优化器 optimizer，这里使用的是 Adam 优化器，用于更新模型的参数。
            bceloss = nn.BCELoss()  # 定义了一个二元交叉熵损失函数 bceloss，用于计算模型预测值和真实标签之间的损失。

            """
            开始训练循环

            外层循环是对于 args.Epoch 次数的训练迭代。在每个迭代中，首先将模型设置为训练模式 model.train()

            内层循环遍历训练数据集 dataset_train 中的每个批次数据 batch，并对模型进行训练。
            首先将批次数据移动到指定设备 device 上，然后将优化器的梯度归零，进行前向传播和反向传播计算损失，并更新模型参数。
            """
            for e in range(args.Epoch):
                excute_time = time.asctime()
                model = model.train()
                for batch_idx, batch in enumerate(dataset_train):  # 遍历了(enumerate 枚举) 训练数据集中的每个批次（batch_size=64）数据
                    data_prot = batch.to(device)
                    optimizer.zero_grad()
                    output = model(data_prot)
                    loss = bceloss(output, data_prot.label)
                    loss.backward()
                    optimizer.step()
                    # a = model(batch)

                """
                在每个训练周期结束后，将模型设置为评估模式 model.eval()，然后在验证数据集 dataset_valid 上进行验证。
                通过将每个批次的数据传递给模型，并计算模型输出与真实标签之间的损失。在验证过程中，不进行梯度更新，因此使用 torch.no_grad() 上下文管理器。
                """
                model = model.eval()
                total_preds = torch.Tensor()
                total_labels = torch.Tensor()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataset_valid):  # 遍历了(enumerate 枚举) 验证数据集中的每个批次数据
                        data_prot = batch.to(device)
                        output_valid = model(data_prot)
                        total_preds = torch.cat((total_preds, output_valid.cpu()), 0)
                        total_labels = torch.cat((total_labels, data_prot.label.cpu()), 0)
                    loss_valid = bceloss(total_preds, total_labels)

                """
                在每个验证周期结束后，评估模型性能，并根据评估结果保存最佳模型。
                如果当前的 m-AUPR（平均准确率-召回率）结果比之前的最佳结果更好，则保存当前模型，并将 m-AUPR 更新为当前值。
                """
                perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())
                if perf['all']['m-aupr'] > m_AUPR:
                    m_AUPR = perf['all']['m-aupr']
                    # torch.save(model.state_dict(),'./data_collect/'+str(func)+'/model/'+str(data_source)+'_'+str(func)+'_'+args.net+'_'+str(args.hidden_dim)+'_'+str(m_AUPR)+'_'+args.pool+'_'+str(args.dropout)+'_'+str(args.threshold)+'.pkl')

                    # AUPR结果比之前好 保存这个epoch训练好的模型
                    # ../UltraDataset/Database1_Uniprot+PDB
                    torch.save(model.state_dict(),
                               '../UltraDataset/Database1_Uniprot+PDB/' + str(func) + '/model/' + str(data_source) + '_' + str(
                                   func) + '_' + args.net + '_' + str(args.hidden_dim) + '_' + args.pool + '_' + str(
                                   args.dropout) + '_' + str(args.threshold) + '.pkl')
                    if args.save_results:
                        with open('../UltraDataset/Database1_Uniprot+PDB/' + str(func) + '/' + str(data_source) + '_' + str(func) + '.json',
                                  'w') as f:
                            json.dump(perf, f)
                excute_end_time = time.asctime()
                print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss.cpu().detach().numpy(), '\tValid loss:\t',
                      loss_valid.numpy(), '\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],
                      '\tF-max:\t', perf['all']['F-max'],'\nexcute start from:',excute_time,'\nexcute end in:',excute_end_time)
                # print(total_preds.shape,total_labels.shape)

            # perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())


'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--func', type=lambda s: [item for item in s.split(",")],
                        default=['mf','bp','cc'], help="list of func to predict.")#,'bp','cc'
    parser.add_argument('--data', type=lambda s: [item for item in s.split(",")],
                        default=['pdb', 'alpha'], help="data source.")#, 'alpha'
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help="learning rate.")
    parser.add_argument('--Epoch', type=int,
                        default=50, help="epoch for training.")
    parser.add_argument('--save_results', type=int, default=0, help="whether to save the performance results")
    parser.add_argument('--save_model', type=int, default=1, help="whether to save the model parameters")
    parser.add_argument('--net', type=str, default='GCN', help="GCN or GAT for model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dim for linear")
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda for model")
    parser.add_argument('--pool', type=str, default='gap-gmp', help="pool for model(gep、gap、gmp、gap-gep、gap-gmp)")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for model")
    parser.add_argument('--threshold', type=float, default=8.0, help="distance threshold between residues")
    args = parser.parse_args()
    print(args)
    train(args)

'''

# test run on refined PDB-B dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--func', type=lambda s: [item for item in s.split(",")],
                        default=['mf'], help="list of func to predict.")  # ,'bp','cc'
    parser.add_argument('--data', type=lambda s: [item for item in s.split(",")], default=['pdb'], help="data source.")  # , 'alpha'
    parser.add_argument('--learning_rate', type=float,default=0.0001, help="learning rate.")
    parser.add_argument('--Epoch', type=int, default=50, help="epoch for training.") #默认50，也可以设定为30
    parser.add_argument('--save_results', type=int, default=0, help="whether to save the performance results")
    parser.add_argument('--save_model', type=int, default=1, help="whether to save the model parameters")
    parser.add_argument('--net', type=str, default='combine_net', help="GCN or GAT or GCN_GCNConv+GAT_GATv2Conv or GCN_SAGEConv+GAT_GATv2Conv or combine_net for model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dim for linear")  # 原论文中隐藏层设定为512，在这个实验先把它设定为256效果不好，最后还是改回512
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda for model")
    parser.add_argument('--pool', type=str, default='gap-gmp', help="pool for model(gep、gap、gmp、gap-gep、gap-gmp)")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for model")
    parser.add_argument('--threshold', type=float, default=8.0, help="distance threshold between residues")
    args = parser.parse_args()
    print(args)
    train(args)




