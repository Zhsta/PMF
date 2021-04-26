# pmf编程作业
学号：2020223040005<br>
姓名：赵小明<br>
u.data是MovieLens 100k数据集<br>
pmf.py 一共有三个功能函数load_data,rmse,fit.<br>
load_data用来载入MovieLens 100k数据集，并且根据传入的ratio参数，将数据集分为训练集和测试集<br>
rmse用来计算当前的U,V矩阵与rating的rmse<br>
fit函数接收设定好的参数、训练集和测试集，进行训练，并记录每一个epoch的loss和rmse<br>
运行截图如下：<br>
![image](https://user-images.githubusercontent.com/34116348/114387369-eef71e80-9bc4-11eb-84f5-f6ea22893959.png)
![image](https://user-images.githubusercontent.com/34116348/114387397-f61e2c80-9bc4-11eb-91d0-f86e60ec930f.png)

# LARA编程作业
# LARA

data_loads.py是工具文件，主要用来载入数据和计算评价指标的函数<br>
csv文件是在movie_lens数据集上已经处理好的数据<br>
new.py是模型文件，在train函数中进行训练，每训练一轮计算一次评价指标<br>
运行结果如下：<br>
![2](https://user-images.githubusercontent.com/34116348/116082297-d4956880-a6cd-11eb-8b34-be3fbf9ecc98.JPG)
![image](https://user-images.githubusercontent.com/34116348/116082329-de1ed080-a6cd-11eb-8537-0d8cc80cedeb.png)

