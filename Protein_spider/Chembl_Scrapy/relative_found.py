activities_id_list = ['31863', '31864', '31865', '31866', '31867']
relate_list_molecule = ['CHEMBL113081', 'CHEMBL324340', 'CHEMBL324340', 'CHEMBL109600', 'CHEMBL109600']
relate_list_target = ['CHEMBL1806', 'CHEMBL3921', 'CHEMBL3879801', 'CHEMBL3921', 'CHEMBL3879801']
relate_list_assay = ['CHEMBL663853', 'CHEMBL872937', 'CHEMBL693237', 'CHEMBL872937', 'CHEMBL693238']
relate_list_document = ['CHEMBL1137930', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658']
relative_list=[activities_id_list,relate_list_molecule,relate_list_target,relate_list_assay,relate_list_document]
#将所有的关系放在一起，做成一个二维列表，方便对比和索引
#print('test:',relative_list[4][3])
#act_rela_molecule
# k=0
# i=1
# j=0
# relate_dict = {}
# cache = ''
# while(j <= len(relate_list_molecule)-1 ):
#     str1 = relative_list[k][j] #遍历二维数组第一行 即activities_id_list
#     str2 = relative_list[i][j] #遍历二维数组第二行 即relate_list_molecule
#     if (str2 != cache):
#         relate_dict[str(str2)] = str1
#         cache = str2
#     else:#cache 与下面的遍历一至说明molecule id 重复了
#         relate_dict[str(str2) + str(cache)] = str1
#         cache = str2
#     j += 1
# print(relate_dict)

'''
activities_id_list = ['31863', '31864', '31865', '31866', '31867']
relate_list_molecule = ['CHEMBL113081', 'CHEMBL324340', 'CHEMBL324340', 'CHEMBL109600', 'CHEMBL109600']
'''

#flag = 0
relate_dict = {}
list1 = []  #用来存放没有重复的molecule id 对应的activities_id（一个列表就存唯一一个activities_id）
list2 = []  # 用来存放该重复的molecule id对应的不同activities_id（一个列表存多个activities_id）
j=0
cache = ''
while(j <= len(relate_list_molecule)-1 ):
    flag = 0
    str1 = relative_list[0][j] #遍历二维数组第一行 即activities_id_list
    str2 = relative_list[1][j] #遍历二维数组第二行 即relate_list_molecule
    while (j+1 <= len(relate_list_molecule)-1 and str2 == relative_list[1][j+1]): #此时的molecule id与下一个molecule id重复
        list2.append(str1)
        list2.append(relative_list[0][j+1])
        # print("此时的molecule id：",str2,"他的list2=",list2)
        j += 1
        flag = 1
    #上面这个while循环结束后会得到一个列表存放当前重复的molecule id对应的所有activities_id
    if(flag == 1): #flag=1 说明发生过上面的while循环
        # print("这个时候的molecule id：", str2, "他的list2=", list2)
        relate_dict[str2] = list2 # 存入字典关键字为当下的molecule id，值为list2内所有的值。
        #print('dict demo:',relate_dict)
        list2 = []
        j += 1
    elif (j+1 == len(relate_list_molecule)-1):#已经遍历到队尾了尽管此时flag=0 但是它任然是重复的，上面已经插过了不需要再插入了。
        break
    else: #当前molecule id没与下一个发生重复,且没到队尾
        list1.append(str1)
        relate_dict[str2] = list1
        list1 = []
        j += 1
print(relate_dict)



    # while(str2 == cache and j <= len(relate_list_molecule)-1):#对重复部分进行处理
    #     # if (str2 != cache):
    #     #     list1.append(str1)
    #     #     cache = str2
    #     # else:#cache 与下面的遍历一至说明molecule id 重复了
    #     #     relate_dict1[str(str2) + str(cache)] = str1
    #     #     cache = str2
    #     list2.append(str1)
    #     j += 1
    #     flag = 1
    # cache = str2
    # if (flag == 1):#flag = 1 说明出现了重复。
    #     relate_dict1[str2] = list2
    #     list2 = []#将list2 重新置空
    #     j -= 1 #while 里面加了1了，后面还要+1，因此把这个1减掉
    # else:#没出现molecule id重复
    #     if(str2 == relative_list[1][j+1]):#判断下一个id是否相同
    #
    #     list1.append(str1)
    #     relate_dict1[str2] = list1
    # j += 1




def find_relation(a,b):#a,b 分别代表要比对的是哪两行
    relate_dict = {}
    list1 = []  # 用来存放没有重复的molecule id 对应的activities_id（一个列表就存唯一一个activities_id）
    list2 = []  # 用来存放该重复的molecule id对应的不同activities_id（一个列表存多个activities_id）
    j = 0
    #cache = ''
    while (j <= len(relate_list_molecule) - 1):
        flag = 0
        str1 = relative_list[a][j]  # 遍历二维数组第a行 即activities_id_list
        str2 = relative_list[b][j]  # 遍历二维数组第b行 即relate_list_molecule
        while (j + 1 <= len(relate_list_molecule) - 1 and str2 == relative_list[1][
            j + 1]):  # 此时的molecule id与下一个molecule id重复
            list2.append(str1)
            list2.append(relative_list[0][j + 1])
            # print("此时的molecule id：",str2,"他的list2=",list2)
            j += 1
            flag = 1
        # 上面这个while循环结束后会得到一个列表存放当前重复的molecule id对应的所有activities_id
        if (flag == 1):  # flag=1 说明发生过上面的while循环
            # print("这个时候的molecule id：", str2, "他的list2=", list2)
            relate_dict[str2] = list2  # 存入字典关键字为当下的molecule id，值为list2内所有的值。
            print('dict demo:', relate_dict)
            list2 = []
            j += 1
        elif (j + 1 == len(relate_list_molecule) - 1):  # 已经遍历到队尾了尽管此时flag=0 但是它任然是重复的，上面已经插过了不需要再插入了。
            break
        else:  # 当前molecule id没与下一个发生重复,且没到队尾
            list1.append(str1)
            relate_dict[str2] = list1
            list1 = []
            j += 1
    return relate_dict

dict1 = find_relation(0,3)
#print("dict1 =",dict1)


'''
relative with target
activities_id_list = ['31863', '31864', '31865', '31866', '31867']
relate_list_target = ['CHEMBL1806', 'CHEMBL3921', 'CHEMBL3879801', 'CHEMBL3921', 'CHEMBL3879801']
'''



i = 0
j = 1
relate_dict_index = {}
relate_dict = {}
cache = []#存储指针已经走过的重复的位置，不需要再走一遍了
position = []
while(i <= len(relative_list[2])-1):#relative_list[2] == relate_list_target
    position = [relative_list[0][i]]
    j = i+1
    while(j <= len(relative_list[2]) -1):
        if(relative_list[2][i] == relative_list[2][j]):
            # position.append(i)
            position.append(relative_list[0][j])
            cache.append(j)
        j += 1
    relate_dict_index[relative_list[2][i]] = position
    i += 1
    while(i in cache): #cache 中已经保存的位置说明这个位置不需要再进行比对了。
        i += 1
#print('relative found:',relate_dict_index)


# for keyword in relate_dict_index:
#     print('test22222:',keyword)
#     #relate_dict[keyword] = relate_dict_index[keyword]
#     for value in relate_dict_index[keyword]:
#         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]

#
# def act_relative_find(a):
#     i = 0
#     j = 1
#     relate_dict_index = {}
#     cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
#     position = []
#     while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
#         position = [relative_list[0][i]]
#         j = i + 1
#         while (j <= len(relative_list[a]) - 1):
#             if (relative_list[a][i] == relative_list[a][j]):
#                 # position.append(i)
#                 position.append(relative_list[0][j])
#                 cache.append(j)
#             j += 1
#         relate_dict_index[relative_list[a][i]] = position
#         i += 1
#         while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
#             i += 1
#     return relate_dict_index
#     # for keyword in relate_dict_index:
#     #     print('test22222:',keyword)
#     #     #relate_dict[keyword] = relate_dict_index[keyword]
#     #     for value in relate_dict_index[keyword]:
#     #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]
#
# print('relative found:',act_relative_find(2))
#
#
# def whatever_relative_find(b,a):
#     i = 0
#     j = 1
#     relate_dict_index = {}
#     cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
#     position = []
#     while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
#         position = [relative_list[b][i]]
#         j = i + 1
#         while (j <= len(relative_list[a]) - 1):
#             if (relative_list[a][i] == relative_list[a][j]):
#                 # position.append(i)
#                 position.append(relative_list[b][j])
#                 cache.append(j)
#             j += 1
#         relate_dict_index[relative_list[a][i]] = position
#         i += 1
#         while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
#             i += 1
#     return relate_dict_index
#     # for keyword in relate_dict_index:
#     #     print('test22222:',keyword)
#     #     #relate_dict[keyword] = relate_dict_index[keyword]
#     #     for value in relate_dict_index[keyword]:
#     #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]
#
# print('whatever relative found:',whatever_relative_find(1,2))


'''
activities_id_list = ['31863', '31864', '31865', '31866', '31867']
relate_list_molecule = ['CHEMBL113081', 'CHEMBL324340', 'CHEMBL324340', 'CHEMBL109600', 'CHEMBL109600']
relate_list_target = ['CHEMBL1806', 'CHEMBL3921', 'CHEMBL3879801', 'CHEMBL3921', 'CHEMBL3879801']
relate_list_assay = ['CHEMBL663853', 'CHEMBL872937', 'CHEMBL693237', 'CHEMBL872937', 'CHEMBL693238']
relate_list_document = ['CHEMBL1137930', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658']
relative_list=[activities_id_list,relate_list_molecule,relate_list_target,relate_list_assay,relate_list_document]
'''

'''
act_relative_find(a) 函数
用来寻找所有四个列表
relate_list_molecule = ['CHEMBL113081', 'CHEMBL324340', 'CHEMBL324340', 'CHEMBL109600', 'CHEMBL109600']
relate_list_target = ['CHEMBL1806', 'CHEMBL3921', 'CHEMBL3879801', 'CHEMBL3921', 'CHEMBL3879801']
relate_list_assay = ['CHEMBL663853', 'CHEMBL872937', 'CHEMBL693237', 'CHEMBL872937', 'CHEMBL693238']
relate_list_document = ['CHEMBL1137930', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658', 'CHEMBL1146658']
与活性物列表：
activities_id_list = ['31863', '31864', '31865', '31866', '31867']
的关联关系
传参a是用来确定relative_list[a] 
用他来确定是要返回化合物表relate_list_molecule和activities_id_list的关系则a=1 因为 relate_list_molecule属于二维列表relative_list的第二行
第三行 a =2 则是返回 target靶点蛋白表relate_list_target和activities_id_list的关系
最后的返回值是一个关系字典如下 是返回化合物表relate_list_molecule和activities_id_list的关系
{'CHEMBL113081': ['31863'], 'CHEMBL324340': ['31864', '31865'], 'CHEMBL109600': ['31866', '31867']}
keyword 键'CHEMBL324340' 是molecule id 
value 值['31864', '31865'] 是对应molecule id 的活性物id
'''
def act_relative_find(a):
    i = 0
    j = 1
    relate_dict_index = {}
    cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
    position = []
    while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
        position = [relative_list[0][i]]
        j = i + 1
        while (j <= len(relative_list[a]) - 1):
            if (relative_list[a][i] == relative_list[a][j]):
                # position.append(i)
                position.append(relative_list[0][j])
                cache.append(j)
            j += 1
        relate_dict_index[relative_list[a][i]] = position
        i += 1
        while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
            i += 1
    return relate_dict_index
    # for keyword in relate_dict_index:
    #     print('test22222:',keyword)
    #     #relate_dict[keyword] = relate_dict_index[keyword]
    #     for value in relate_dict_index[keyword]:
    #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]
print('relative found:',act_relative_find(2))
'''
whatever_relative_find(b,a): 作为def act_relative_find(a): 的升级版本
现在的whatever_relative_find(b,a): 
函数可以传两个任意的参数遍历选择 relative_list=[activities_id_list,relate_list_molecule,relate_list_target,relate_list_assay,relate_list_document]
内的任意两行进行关联比对
如要是想得到relate_list_molecule 化合物 和 relate_list_target 靶点蛋白之间的id关联关系字典 只需要index_dict = whatever_relative_find(1,2) 
即二维列表relative_list 的第二行molecule和第三行target 进行比对
返回的index_dict = {'CHEMBL1806': ['CHEMBL113081'], 'CHEMBL3921': ['CHEMBL324340', 'CHEMBL109600'], 'CHEMBL3879801': ['CHEMBL324340', 'CHEMBL109600']}
keyword 为target id 
value 为不同的target八点蛋白对应的小分子化合物
'''
def whatever_relative_find(b,a):
    i = 0
    j = 1
    relate_dict_index = {}
    cache = []  # 存储指针已经走过的重复的位置，不需要再走一遍了
    position = []
    while (i <= len(relative_list[a]) - 1):  # relative_list[2] == relate_list_target
        position = [relative_list[b][i]]
        j = i + 1
        while (j <= len(relative_list[a]) - 1):
            if (relative_list[a][i] == relative_list[a][j]):
                # position.append(i)
                position.append(relative_list[b][j])
                cache.append(j)
            j += 1
        relate_dict_index[relative_list[a][i]] = position
        i += 1
        while (i in cache):  # cache 中已经保存的位置说明这个位置不需要再进行比对了。
            i += 1
    return relate_dict_index
    # for keyword in relate_dict_index:
    #     print('test22222:',keyword)
    #     #relate_dict[keyword] = relate_dict_index[keyword]
    #     for value in relate_dict_index[keyword]:
    #         relate_dict[keyword] = relative_list[0][relate_dict_index[keyword][value]]

print('whatever relative found:',whatever_relative_find(0,4))





















