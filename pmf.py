import numpy as np
import random
import matplotlib.pyplot as plt


def load_data(filename, ratio=0.8):
    user_dict = {}
    item_dict = {}
    N = 0
    M = 0
    u_index = 0
    i_index = 0
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            a = line.split()
            userId = a[0]
            itemId = a[1]
            rating = a[2]
            time = a[3]
            if int(userId) not in user_dict:
                user_dict[int(userId)] = u_index
                u_index += 1
            if int(itemId) not in item_dict:
                item_dict[int(itemId)] = i_index
                i_index += 1
            data.append([user_dict[int(userId)], item_dict[int(itemId)], int(rating)])

    N = u_index
    M = i_index

    np.random.shuffle(data)
    train_data = data[0:int(len(data)*ratio)]
    test_data = data[int(len(data)*ratio):]

    return N, M, train_data, test_data




def RMSE(U,V,test):
    num = len(test)
    sum_rmse = 0.0
    for t in test:
        u = t[0]
        i = t[1]
        r = t[2]
        pr = np.dot(U[u], V[i].T)
        sum_rmse += np.square(r-pr)
    rmse = np.sqrt(sum_rmse/num)
    return rmse


def fit(train_set, test_set, N, M, D, lr, lambda_u, lambda_v, max_epoch):
    #均值为0 标准差为0.1
    U = np.random.normal(0, 0.1, (N, D))
    V = np.random.normal(0, 0.1, (M, D))
    rmse_list = []
    loss_list = []
    for epoch in range(1,max_epoch+1):
        los = 0.0
        for data in train_set:
            userId = data[0]
            itemId = data[1]
            rating = data[2]

            e = rating - np.dot(U[userId], V[itemId].T)
            U[userId] = U[userId] + lr*(e*V[itemId]-lambda_u*U[userId])
            V[itemId] = V[itemId] + lr*(e*U[userId]-lambda_v*V[itemId])

            los = los+e**2
            #+lambda1*np.square(U[userId]).sum()+lambda2*np.square(V[itemId]).sum())
        loss1 = 0
        loss2 = 0
        for i in range(N):
            loss1 += np.sqrt(np.square(U[i]).sum())
        loss1 = lambda_u*loss1
        for j in range(M):
            loss2 += np.sqrt(np.square(V[j]).sum())
        loss2 = lambda_v*loss2
        los = 0.5*(los+loss1+loss2)
        loss_list.append(los)
        rmse = RMSE(U, V, test_set)
        rmse_list.append(rmse)
        print('epoch:'+str(epoch)+'   loss:'+str(los)+'   rmse:'+str(rmse))

    return U, V, loss_list, rmse_list

if __name__ == '__main__':
    filename = 'u.data'
    N, M, train_set, test_set = load_data(filename)
    lr = 0.005
    lambda_u = 0.1
    lambda_v = 0.1
    D = 10
    max_epoch = 50
    U, V, loss_list, rmse_list = fit(train_set,test_set,N,M,D,lr,lambda_u,lambda_v,max_epoch)
    print(loss_list)
    print(rmse_list)
