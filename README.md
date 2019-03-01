本系统是一个医疗问答系统，针对全科问题，基于知识图谱和问答对数据库实现的。
# 系统架构图 #

![系统架构图](https://raw.githubusercontent.com/zhihao-chen/MedicalQASystem/master/img/%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%84%E5%9B%BE.png)
# 系统流程图 #
![系统流程图](https://raw.githubusercontent.com/zhihao-chen/MedicalQASystem/master/img/%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%841.png)
# 系统效果图 #
![系统效果图](https://raw.githubusercontent.com/zhihao-chen/MedicalQASystem/master/img/3.png)
# 知识图谱结构 #
![知识图谱结构](https://raw.githubusercontent.com/zhihao-chen/MedicalQASystem/master/img/%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1.png)
# 运行 #


1.首先你需要建立知识图谱，存储在neo4j中。
在DATA目录下:python build_graph.py
2.建立倒排索引，存在MySQL中
在DATA目录下：python bool_retrive.py

还有一些细节就不讲了。
python app.py

# 不足之处 #


1. 知识图谱和问答对数据库都在小了，能回答的问题不是很多。
2. 意图识别分类模型是个多分类模型，今后将训练多标签分类模型。但训练数据很难获得，自己标记太累了。
3. 问答匹配模型效果还不是很好，需要继续优化。
4. 网页界面不是很美观。哈哈。
5. 其它的不足想起来了再加.... 

 联系方式：andrew_czh@163.com
