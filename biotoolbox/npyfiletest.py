import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

array1 = np.load('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/contact_map_6582_8.0/1A0H-A.npy')
array2 = np.load('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/contact_map_6582_8.0/5XYI-a.npy')
array3 = np.load('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/contact_map_6582_8.0/5XYI-g.npy')
array4 = np.load('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/contact_map_6582_8.0/6AZ1-C.npy')
array5 = np.load('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/contact_map_6582_8.0/5XXU-B.npy')
data = np.load('../Protein_Predict/DeepFRI-master/examples/pdb_cmaps/1S3P-A.npz')

print("1A0H-A.npy:",array1,'\n')
print("5XYI-a.npy:",array2,'\n')
print("5XYI-g.npy:",array3,'\n')
print("6AZ1-C.npy:",array4,'\n')
print("5XXU-B.npy:",array5,'\n')
'''打印npz文件中的键（keys）确定其中包含数组的名称'''
print("keys in the compare_array 1S3P-A.npz file:",data.files,'\n') #['C_alpha', 'C_beta', 'seqres']
#接下来提取需要做可视化的数组，如果数组是图像数据，可以使用matplotlib进行可视化
npz_arry1 = data['C_alpha']
npz_arry2 = data['C_beta']
npz_arry3 = data['seqres']

print('npz_arry1即 C_alpha：',npz_arry1,'\n')
print('npz_arry2即 C_beta：',npz_arry2,'\n')
print('npz_arry3即 seqres：',npz_arry3,'\n')

plt.figure(figsize=(10,5)) #这行代码创建了一个新的图形窗口，并指定了该图形的大小为宽度10英寸、高度5英寸。
plt.subplot(1, 3, 1)
'''
plt.subplot(1, 2, 1) 
这行代码创建了一个包含 1 行 3 列的子图网格，并选择了第一个子图来绘制。
参数 (1, 3, 1) 指定了子图网格的布局，其中 (1, 3) 表示总共有1行3列子图，而 ，1） 表示当前选中的是第一个子图。
'''
plt.imshow(npz_arry1, cmap='gray')
'''
plt.imshow(npz_arry1, cmap='gray')
这行代码使用 imshow() 函数将名为 npz_arry1 的 NumPy 数组绘制成图像 其中:
- npz_arry1 是一个二维数组，通常是表示图像的像素值矩阵。
- imshow() 函数用于将二维数组中的数值映射为颜色，然后以图像的形式显示出来。
- cmap='gray' 指定了使用灰度色彩映射（colormap），即将二维数组中的值映射到灰度颜色空间中，显示出黑白图像。这意味着数值较低的点将显示为较暗的灰色，而数值较高的点将显示为较亮的灰色。
'''
plt.title('npz_arry1: C_alpha')

plt.subplot(1, 3, 2) # 选择第二个子图进行绘制
plt.imshow(npz_arry2, cmap='viridis') # viridis 色彩映射，它是一种颜色渐变映射。
plt.title('npz_arry2: C_beta')

plt.subplot(1, 3, 3)
plt.imshow(array3, cmap='viridis') # cmap='jet' 指定了使用 jet 色彩映射，它是一种带有彩虹色彩的映射，适用于显示多种颜色。
plt.title('5XYI-g.npy')


plt.show()


'''
结论：
.npy 和 .npz 都是与 NumPy 库相关的文件扩展名，用于存储 NumPy 数组数据。它们之间的主要区别在于存储方式和文件结构：

.npy 文件：

.npy 文件是 NumPy 的二进制格式，用于存储单个 NumPy 数组。
它以二进制格式存储数组数据，并且可以保留数组的结构、形状、dtype 等信息。
由于是单个数组，.npy 文件适用于存储单个数组或加载单个数组。
.npz 文件：

.npz 文件也是 NumPy 的二进制格式，但是它可以存储多个 NumPy 数组。
.npz 文件实际上是一个压缩文件，其中包含了多个 .npy 格式的数组以及数组的名称。
这种格式适用于需要同时保存和加载多个 NumPy 数组的情况，例如在实验数据处理中，通常会产生多个相关联的数组。
综上所述，.npy 文件用于单个数组的存储，而 .npz 文件用于存储多个相关联的数组，并且能够将它们以压缩格式存储在一个文件中，方便管理和传输。

'''


'''
对npz 文件分析后可得出结论，每个npz文件包含三个部分:
keys in the compare_array 1S3P-A.npz file: ['C_alpha', 'C_beta', 'seqres'] 
其中C_alpha 和 C_beta 都为npy文件即numpy数组
而seqres 则为一字符串文件 代表当前如 1S3P-A.npz 对应的蛋白质序列
seqres： SMTDLLSAEDIKKAIGAFTAADSFDHKKFFQMVGLKKKSADDVKKVFHILDKDKDGFIDEDELGSILKGFSSDARDLSAKETKTLMAAGDKDGDGKIGVEEFSTLVAES 




'''

# def persistent_load(pers_id):
#     return None
#
# with open('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/model/bp/30/pdb_amplifed_bp_GCN_512_gap-gmp_0.2_8.0.pkl','rb') as file1:
#     unpickler= pk.Unpickler(file1)
#     unpickler.persistent_load = persistent_load()
#
#     loaded_data = unpickler.load()
#
# with open('G:/fkfk/Struct2Go-main/data_collect/amplify_samples/model/bp/30/pdb_no_amplifed_bp_GCN_512_gap-gmp_0.2_8.0.pkl','rb') as file2:
#     unpickler = pk.Unpickler(file2)
#     unpickler.persistent_load = persistent_load()
#
#     loaded_data1 = unpickler.load()
#
# print("pdb_amplifed_bp_GCN_512_gap-gmp_0.2_8.0.pkl = ",loaded_data)
# print("pdb_no_amplifed_bp_GCN_512_gap-gmp_0.2_8.0.pkl = ",loaded_data1)


