import pymysql
connection_flag = 0
while(connection_flag == 0):
    try:
        db = pymysql.connect(host='localhost', user='root', passwd='Qaz123456', port=3306)
        print('连接成功！')
        connection_flag = 1
    except:
        print('something wrong! database connection failed')
cursor = db.cursor()# 使用 cursor() 方法创建一个游标对象 cursor
cursor.execute("SELECT VERSION()")#使用 execute()  方法执行 SQL 查询
data = cursor.fetchone() #使用fetchone()方法获取单条数据
print("Database version = %s"%data)

