'''该代码用来复现一些简单的pymongo 操作'''
from pymongo import MongoClient
import copy
from bs4 import BeautifulSoup
import urllib.request as request #urllib3版本要<=1.25否则会与requests版本冲突


#print(time.localtime())#当前时间
mongoClient = MongoClient('mongodb://localhost:27017/')
dtmr_dev = mongoClient.get_database('pymongo_test')
dev = dtmr_dev.get_collection('human')

'''插入数据'''
# #插一条
# person_dict = {'name':'lisheng','age':23,'gender':'male'}
# result = dev.insert_one(person_dict)
#插多条
person_dict = {'name':'lisheng','age':24,'gender':'male'}
person_dict1 = {'name':'zhangsan','age':15,'gender':'female'}
person_dict2 = {'name':'lisi','age':14,'gender':'male'}
person_dict3 = {'name':'wangwu','age':13,'gender':'male'}
person_list = [person_dict1,person_dict2,person_dict3]
person_dict1 = {'name':'zhangsan','age':100,'gender':'male'}
person_list2 = [person_dict,person_dict1,person_dict2,person_dict3]
# repeat_dict3 = {'name':'wangwu','age':13,'gender':'male'} #插入重复内容并不会覆盖相同内容
#dev.insert_many(person_list)

#更新，通过设置update第三个参数为True，再设置第一个参数为（每条记录的）唯一值对每条记录进行过滤，匹配到相同的唯一值，进行更新，没有匹配到，则插入。达到插入前进行去重的效果。
for element in person_list2:
    temp = dev.update_one({'name': element['name']}, {'$set': element}, True)
    print(temp)
count = dev.find({"age":124})
print("count=",count)
num = dev.count_documents({"age":124})
print(num)
























#
# i = 0
# limit=input("你要几条activities id？？")
# start_point=input('从第几个数据开始？？')
# html_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit="+str(limit)+"&offset="+str(start_point)#直接从start_point处拿chembl网站给的前limit条activities数据
# '''
# 发现chembl网页 有断点功能：
# https://www.ebi.ac.uk/chembl/api/data/activity?limit=1&offset=2
# limit 是网页接口响应后传回的数据条数限制，offset相当于chembl网页的断点 如：
# limit = 20
# offset = 1
# 说明接口会从第1条数据开始返回，返回数据条数为20条
# ['31863', '31864', '31865', '31866', '31867', '31868', '31869', '31870', '31871', '31872', '31873', '31874', '31875', '31876', '31877', '31878', '31879', '31880', '31881', '31882']
# 若
# limit = 10
# offset = 10
# 返回值就为
# ['31873', '31874', '31875', '31876', '31877', '31878', '31879', '31880', '31881', '31882']
#
# '''
# activities_id_list = []
# response = request.urlopen(html_url)
# XML = response.read()
# soup = BeautifulSoup(XML,'xml')
# all_activities = soup.activities.find_all('activity',recursive=False)
#
# while(i <= len(all_activities)-1 ):
#     activities_id_list.append(all_activities[i].activity_id.string)
#     i += 1
# print(activities_id_list)

