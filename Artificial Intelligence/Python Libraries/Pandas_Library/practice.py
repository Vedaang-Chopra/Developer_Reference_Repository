# # import numpy as np
# import csv
# # # print(np.array((([13,35,74,48],[23,37,37,38],[73,39,93,39])))[:,(1,2)])
# #
# # import pandas as pd
# # # score=[10,15,20,25]
# # # print(pd.Series(data=score,index=['a','b','c','d']))
# # #
# # # a=((np.arange(10,21)[0:7])[:])
# # # a[:]=101
# # # print(a)
# # # s1=pd.Series(['a','b'])
# # # s2=pd.Series(['c','d'])
# # # print(pd.concat([s1,s2]))
# #
# # data={'prodID':['101','102','103','104','104'],
# #       'prodname':['X','Y','Z','X','W',],
# #       'profit':['2738','2727','3497','7347','3743',]}
# # print((pd.DataFrame(data).groupby('prodID').max()))
#
# with open('Disease_Definiton.csv', 'r',encoding='utf-8') as csvfile:
#     data = list(csv.reader(csvfile))
# csvfile.close()
# a=[]
# for i in range(1,len(data)):
#     a.append(data[i][2])
# print(a[0])
# print(len(a))
# b=['autosomal recessive disorder','cough','fever']
# # with open('submitting_report.csv', 'r') as csvfile:
# #     data = list(csv.reader(csvfile))
# # csvfile.close()
#
# # b=['pain chest','cough','fever']
# c=[]
# temp=[]
# for i in range(0,len(a)):
#     temp=[]
#     for j in range(0,len(b)):
#         if (a[i].__contains__(b[j]))==True:
#             temp.append(b[j])
#     c.append(temp)
# print(c)

