import pandas as pd
data = pd.read_csv('uniprot_id.csv',engine='python')#sep='/n',
print(type(data))#<class 'pandas.core.frame.DataFrame'>
#Pandas DataFrame 是一个二维的数组结构，类似二维数组。
print(data)
# print('索引=',data.index)
print('返回第一行，第一列=',data.loc[0][0])
print('返回第一行，第二列=',data.loc[0][1])
# print('返回第一行和第二行所有第一列的值=',data.loc[[0,1][0]])


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

uniprot_id_list,uniprot_name_list = return_id_name_list(data)
print(len(uniprot_id_list))
print(len(uniprot_name_list))
print(uniprot_id_list[20421])







# a = str(data.loc[0]).split(",")
# print(a)

# def listgenerate(i):
#     while



'''
1. 该代码旨在通过pandas 读取uniprot_id.csv文件 后将其切片生成两个列表（uniprot_id_list and uniprot_name_list）

1.1 pandas的dataframe行列索引方法与查询：（参考文献：https://blog.csdn.net/rouge_eradiction/article/details/108586925）
@ dataframe行索引 (dataframe的行索引方法有三种，分别为loc，iloc,ix):
@.1 loc:
loc是基于行索引（index），或者说是行的名称进行索引的。比如如果说有自己认为设置了索引的名称，在进行检索时使用loc，就只能输入行的名称。
但是如果index是默认的递增数，那么和iloc没有区别。要注意此时如果使用切片索引，如[0:k]那么取的是index从0到k的k+1个行，而不是k-1行。
@.2 iloc:
iloc是根据行的序列数索引的，序列数从0开始取，iloc前面的i就好像是在提醒你，他的输入参数是自然数。当然，在仅仅使用数字取列时，可以直接使用df[1]的方法。
@.3 ix:
ix是前两者的混合，输入任何一种都可以。
这三种方式都支持对行列的检索和切片，组合起来可以有很丰富的用法。
@ 列索引：
dataframe的列索引比较简单一般可以使用df['cloumn']的方式解决。


'''