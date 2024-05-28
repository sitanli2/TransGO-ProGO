import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from contact_map_builder import ContactMapContainer,DistanceMapBuilder

# mapbuilder = DistanceMapBuilder
# strtest = "MGLEALVPLAMIVAIFLLLVDLMHRHQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREAMVTRGEDTADRPPAPIYQVLGFGPRSQGVILSRYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFADQAGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLPHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAKKEKAKGSPESSFNDENLRIVVGNLFLAGMVTTSTTLAWGLLLMILHLDVQRGRRVSPGCPIVGTHVCPVRVQQEIDDVIGQVRRPEMGDQAHMPCTTAVIHEVQHFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWKKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVAAGQPRPSHSRVVSFLVTPSPYELCAVPR"
# print(len(strtest)) #验证一下是不是515个氨基酸残基


def contact_mapping(UPID): #测试 UPID = "A0A087X1C5"
    f = open("./TestPDBDataset/"+str(UPID)+".pdb","r")
    D = []
    i = 0
    for a in f.readlines(): #遍历PDB文件的每一行
        b = a.split() #对这一行切片
        if b[0] == "ATOM": # 找到 "ATOM" 打头的行
            if b[2] == "CA": # 找到 "CA" 原子的那一行
                D.append((float(b[6]),float(b[7]),float(b[8]))) # 通过循环提取出每一个 CA 原子的空间三维坐标
                #print(len(D))

    """
    最后在选用的A0A087X1C5.pdb例子中，D共有515个钙元素的三维坐标，说明该PDB文件对应的蛋白质共有515个氨基酸残基
    接下来通过两重循环计算每个氨基酸与任意一个氨基酸CA原子的距离
    """

    distance1 = []
    for b in range(len(D)):
        distance2 = []
        for c in range(len(D)):
            distance = ((D[b][0] - D[c][0]) ** 2 + (D[b][1] - D[c][1]) ** 2 + (D[b][2] - D[c][2]) ** 2) ** 0.5 #其实就是欧氏距离计算公式
            if distance == 0.:
                distance2.append(0)
            elif (distance <= 8. and distance != 0):
                distance2.append(round(float(distance),3)) #round(float(distance), 3) 这段代码是将一个浮点数 distance 四舍五入保留三位小数
            elif (distance > 8.): # 检查distance是否大于8.0
                distance2.append(0)
        distance1.append(distance2)
    #print(distance1)

    """
    接下来将矩阵转换为numpy形式
    """
    distance1 = np.array(distance1) #转为numpy矩阵
    print("value of the generated contact map：\n",distance1,"\n shape of the map：\n",np.shape(distance1))

    """经上面打印后可知，这个contact map 是一个515*515 即n*n的矩阵，我还需要将他转换为符合GCN输入的[Source,Target] 即2*n的形式作为整个GCN的edge_index"""

    Source = []
    Target = []
    """遍历接触图，找到存在接触的节点对"""
    for i in range(distance1.shape[0]): #distance1.shape[0]就是取shape的第一个维度，即515
        for j in range(distance1.shape[1]): #通过两个for循环遍历整个 n*n的矩阵
            if (distance1[i,j] != 0.): # 判断第i行第j列是否为0，不为零则说明第i个氨基酸和第j个氨基酸残基互相接触
                Source.append(i)
                Target.append(j)

    adjacency_matrix = [Source,Target]
    ContactMap = np.array(adjacency_matrix)
    print("符合GCN输入的adjacency_matrix：",ContactMap)
    filepath = "./ContactMapFiles/"+UPID+".npy"
    np.save(filepath,ContactMap)

    # 加载保存的 .npy 文件，以验证保存是否成功
    loaded_data = np.load(filepath)
    print("Loaded data:\n",loaded_data)



    """通过networkx 和 matplotlib进行可视化"""
    # G = nx.Graph()
    #
    # nx.from_numpy_matrix(distance1)
    # nx.draw(G)
    # plt.show()
    # nx.betweenness_centrality(G)
    # nx.closeness_centrality(G)
    # nx.degree_centrality(G)
    # nx.clustering(G)


def UseBiopython():
    """使用PDBparser读取和解析PDB文件"""
    pdb_code = "A0A087X1C5"
    parser = PDBParser()
    structure = parser.get_structure(pdb_code, "./TestPDBDataset/A0A087X1C5.pdb")

    # 计算残基之间的距离
    def calc_residue_dist(residue_one, residue_two):
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def calc_dist_matrix(chain_one, chain_two):
        dist_matrix = np.zeros((len(chain_one), len(chain_two)), dtype=np.float)
        for i, residue_one in enumerate(chain_one):
            for j, residue_two in enumerate(chain_two):
                dist_matrix[i, j] = calc_residue_dist(residue_one, residue_two)
        return dist_matrix

    # 构建接触图
    chain_one = structure[0]["A"]
    chain_two = structure[0]["B"]
    dist_matrix = calc_dist_matrix(chain_one, chain_two)
    contact_map = dist_matrix < 12.0  # 使用阈值确定接触

    # 打印最小和最大距离
    print("Minimum distance:", np.min(dist_matrix))
    print("Maximum distance:", np.max(dist_matrix))
    print("contact map:",contact_map)


if __name__ == '__main__':
    contact_mapping("A0A087X1C5")
    # UseBiopython()