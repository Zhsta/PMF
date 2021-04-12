# pmf编程作业
学号：2020223040005<br>
姓名：赵小明<br>
u.data是数据集<br>
pmf.py 一共有三个功能函数load_data,rmse,fit.<br>
load_data用来载入MovieLens 100k数据集，并且根据传入的ratio参数，将数据集分为训练集和测试集<br>
rmse用来计算当前的U,V矩阵与rating的rmse<br>
fit函数接收设定好的参数、训练集和测试集，进行训练，并记录每一个epoch的loss和rmse<br>
运行截图如下：<br>
![image](https://user-images.githubusercontent.com/34116348/114387369-eef71e80-9bc4-11eb-84f5-f6ea22893959.png)
![image](https://user-images.githubusercontent.com/34116348/114387397-f61e2c80-9bc4-11eb-91d0-f86e60ec930f.png)
