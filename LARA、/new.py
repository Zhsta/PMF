import pandas as pd
import tensorflow as tf
import numpy as np
from LARA import  data_loads
#正则化参数
alpha = 0
#movielens数据集中商品的属性
attribute_num = 18  # the number of attribute
#为属性学习新的向量表示的维度
compress_attribute_num = 5  # the dimention of attribute present
batch_size = 1024
#隐藏层的大小
hidden_layer_dim = 100  # G hidden layer dimention
#用户向量的维度
user_emd_dim = attribute_num

#判别器的参数
discriminator_compress_attribute = tf.get_variable('discriminator_compress_attribute', [2 * attribute_num, compress_attribute_num],
                                 initializer=tf.contrib.layers.xavier_initializer())
discriminator_W1 = tf.get_variable('discriminator_w1', [attribute_num * compress_attribute_num + user_emd_dim, hidden_layer_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
discriminator_b1 = tf.get_variable('discriminator_b1', [1, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
discriminator_W2 = tf.get_variable('discriminator_w2', [hidden_layer_dim, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
discriminator_b2 = tf.get_variable('discriminator_b2', [1, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
discriminator_W3 = tf.get_variable('discriminator_w3', [hidden_layer_dim, user_emd_dim], initializer=tf.contrib.layers.xavier_initializer())
discriminator_b3 = tf.get_variable('discriminator_b3', [1, user_emd_dim], initializer=tf.contrib.layers.xavier_initializer())

discriminator_params = [discriminator_compress_attribute, discriminator_W1, discriminator_b1, discriminator_W2, discriminator_b2, discriminator_W3, discriminator_b3]

#生成器的参数
generator_compress_attribute = tf.get_variable('generator_compress_attribute', [2 * attribute_num, compress_attribute_num],
                                 initializer=tf.contrib.layers.xavier_initializer())
generator_W1 = tf.get_variable('generator_w1', [attribute_num * compress_attribute_num, hidden_layer_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
generator_b1 = tf.get_variable('generator_b1', [1, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
generator_W2 = tf.get_variable('generator_w2', [hidden_layer_dim, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
generator_b2 = tf.get_variable('generator_b2', [1, hidden_layer_dim], initializer=tf.contrib.layers.xavier_initializer())
generator_W3 = tf.get_variable('generator_w3', [hidden_layer_dim, user_emd_dim], initializer=tf.contrib.layers.xavier_initializer())
generator_b3 = tf.get_variable('generator_b3', [1, user_emd_dim], initializer=tf.contrib.layers.xavier_initializer())

generator_params = [generator_compress_attribute, generator_W1, generator_b1, generator_W2, generator_b2, generator_W3, generator_b3]

#要传入的数据
attribute_id = tf.placeholder(shape=[None, attribute_num], dtype=tf.int32)
real_user_emb = tf.placeholder(shape=[None, user_emd_dim], dtype=tf.float32)

neg_attribute_id = tf.placeholder(shape=[None, attribute_num], dtype=tf.int32)
neg_user_emb = tf.placeholder(shape=[None, user_emd_dim], dtype=tf.float32)

#生成器的流程
def generator(attribute_index):
    compressed_attribue_vec = tf.nn.embedding_lookup(generator_compress_attribute, attribute_index)
    flat_compressed_attribute_vec = tf.reshape(compressed_attribue_vec,
                                               shape=[-1, attribute_num * compress_attribute_num])
    layer1 = tf.nn.tanh(tf.matmul(flat_compressed_attribute_vec, generator_W1) + generator_b1)
    layer2 = tf.nn.tanh(tf.matmul(layer1, generator_W2) + generator_b2)
    layer3 = tf.nn.tanh(tf.matmul(layer2, generator_W3) + generator_b3)
    return layer3

#判别器的流程
def discriminator(attribute_index, user_emb):
    compressed_attribute_vec = tf.nn.embedding_lookup(discriminator_compress_attribute, attribute_index)
    flat_compressed_attribute_vec = tf.reshape(compressed_attribute_vec,
                                               shape=[-1, attribute_num * compress_attribute_num])
    concat_att_and_user_emb = tf.concat([flat_compressed_attribute_vec, user_emb], 1)
    layer1 = tf.nn.tanh(tf.matmul(concat_att_and_user_emb, discriminator_W1) + discriminator_b1)
    layer2 = tf.nn.tanh(tf.matmul(layer1, discriminator_W2) + discriminator_b2)
    layer3 = tf.nn.tanh(tf.matmul(layer2, discriminator_W3) + discriminator_b3)
    result = tf.nn.sigmoid(
        layer3
    )
    return layer3, result

#矩阵相乘计算相似度，返回最相似的k个用户的index
def get_k_similiar_user(generator_user, k):
    # print('gu')
    # print(generator_user)
    user_attribute = np.array( pd.read_csv('user_attribute.csv', header=None) )
    user_attribute_matrix = np.transpose(user_attribute)
    similarity_matrix = np.matmul(generator_user, user_attribute_matrix)
    #print(similarity_matrix)
    similarity_user_index = np.argsort(-similarity_matrix)
    return similarity_user_index[:, 0:k]
#对当前生成的用户的向量，计算评价指标
def evaluate(test_data_item, generator_user):
    user_item_interaction = np.array(pd.read_csv('ui_matrix.csv',header=None))
    k = 20
    item_size = np.size(test_data_item)
    similar_user_index = get_k_similiar_user(generator_user, k)
    # print('sui')
    # print(similar_user_index)
    correct_num = 0
    sum = 0.0
    RS = []
    for item, user_list in zip(test_data_item, similar_user_index):
        r = []
        for user in user_list:
            r.append(user_item_interaction[user, item])
            if user_item_interaction[user, item] == 1:
                correct_num += 1
        sum = sum + data_loads.ndcg_at_k(r, k, method=1)
        RS.append(r)
    p_at_20 = round(correct_num / (item_size * k), 4)
    m_at_20 = data_loads.mean_average_precision(RS)
    g_at_20 = sum/item_size

    k = 10
    correct_num = 0
    sum = 0.0
    RS = []
    for item, user_list in zip(test_data_item, similar_user_index):
        r = []
        for user in user_list:
            r.append(user_item_interaction[user, item])
            if user_item_interaction[user, item] == 1:
                correct_num += 1
        sum = sum + data_loads.ndcg_at_k(r, k, method=1)
        RS.append(r)
    p_at_10 = round(correct_num / (item_size * k), 4)
    m_at_10 = data_loads.mean_average_precision(RS)
    g_at_10 = sum/item_size

    return p_at_10,p_at_20,m_at_10,m_at_20,g_at_10,g_at_20

#进行训练
def train():
    fake_user_emb = generator(attribute_id)
    discriminator_real, discriminator_logit_real = discriminator(attribute_id, real_user_emb)
    discriminator_fake, discriminator_logit_fake = discriminator(attribute_id, fake_user_emb)

    discriminator_counter, discriminator_logit_counter = discriminator(neg_attribute_id, neg_user_emb)
    # 计算平均值，计算的都是正例用户 所以使用 labels全为一 计算交叉熵
    discriminator_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logit_real, labels=tf.ones_like(discriminator_logit_real)))
    # 因为希望分不出正例和伪造的 全初始化为1
    discriminator_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logit_fake, labels=tf.zeros_like(discriminator_logit_fake)))
    # 因为是负例，标签为0
    discriminator_loss_counter = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logit_counter, labels=tf.zeros_like(discriminator_logit_counter)))
    # 计算l2 loss  output = sum(t**2)/2 t中的每个元素的平方除以2
    discriminator_regular = alpha * (tf.nn.l2_loss(discriminator_compress_attribute) + tf.nn.l2_loss(discriminator_W1) + tf.nn.l2_loss(discriminator_b1) + tf.nn.l2_loss(
        discriminator_W2) + tf.nn.l2_loss(discriminator_b2) + tf.nn.l2_loss(discriminator_W3) + tf.nn.l2_loss(discriminator_b3))
    generator_regular = alpha * (tf.nn.l2_loss(generator_compress_attribute) + tf.nn.l2_loss(generator_W1) +
                         tf.nn.l2_loss(generator_b1) + tf.nn.l2_loss(generator_W2) + tf.nn.l2_loss(generator_b2) + tf.nn.l2_loss(
                generator_W2) + tf.nn.l2_loss(generator_b2) + tf.nn.l2_loss(generator_W3) + tf.nn.l2_loss(generator_b3))
    #gan在编码时的损失函数
    discriminator_loss = (1 - alpha) * (discriminator_loss_real + discriminator_loss_fake + discriminator_loss_counter) + discriminator_regular
    generator_loss = (1 - alpha) * (tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logit_fake, labels=tf.ones_like(discriminator_logit_fake)))) + generator_regular

    discriminator_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(discriminator_loss, var_list=discriminator_params)
    generator_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(generator_loss, var_list=generator_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for it in range(800):
      #设定生成器和判别器训练的次数
        discriminator_range = 1
        generator_range = 1
        for discriminator_it in range(discriminator_range):
            index = 0
            #防止访问negdata越界
            while index < 253236:
                if index + batch_size <= 253236:
                    train_user_batch, train_item_batch, train_attr_batch, train_user_emb_batch = data_loads.get_train_data(
                        index, index + batch_size)
                    counter_user_batch, counter_item_batch, counter_attr_batch, counter_user_emb_batch = data_loads.get_neg_data(
                        index, index + batch_size)
                index = index + batch_size

                _, discriminator_loss_now, fake_us = sess.run([discriminator_solver, discriminator_loss, fake_user_emb],
                                                  feed_dict={attribute_id: train_attr_batch,
                                                             real_user_emb: train_user_emb_batch,
                                                             neg_attribute_id: counter_attr_batch,
                                                             neg_user_emb: counter_user_emb_batch
                                                             })
            print(discriminator_loss_now)

        for generator_it in range(generator_range):
            index = 0
            while index < 253236:
                if index + batch_size <= 253236:
                    train_user_batch, train_item_batch, train_attr_batch, train_user_emb_batch = data_loads.get_train_data(
                        index, index + batch_size)
                index = index + batch_size

                _, generator_loss_now = sess.run([generator_solver, generator_loss], feed_dict={attribute_id: train_attr_batch})
            print(generator_loss_now)
        #每训练一轮输出评价
        if it % 1 == 0:
            print('迭代次数'+str(it+1))
            test_item_batch, test_attribute_vec = data_loads.get_test_data()
            test_generator_user = sess.run(fake_user_emb, feed_dict={attribute_id: test_attribute_vec})
            #        print( test_generator_user[:10])
            p_at_10, p_at_20, m_at_10, m_at_20, g_at_10, g_at_20 = evaluate(test_item_batch, test_generator_user)
            print('p_at_10 :'+str(p_at_10)+'    p_at_20:'+str(p_at_20))
            print('m_at_10 :'+str(m_at_10)+'    m_at_20:'+str(m_at_20))
            print('g_at_10 :'+str(g_at_10)+'    g_at_20:'+str(g_at_20))



if __name__ == '__main__':
    train()
