import numpy as np
import tensorflow as tf
from sklearn.datasets import make_sparse_coded_signal
import scipy.io as sio

def soft_threshold(x, threshold):            
    #sign(x) * max(0, |x| -T)
    return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0)    

def get_data(n_samples=100, n_features=30, n_components=50, n_nonzero_coefs=7):
    Y,A,X = make_sparse_coded_signal(n_samples=n_samples,n_features=n_features,n_components=n_components
                                    ,n_nonzero_coefs=n_nonzero_coefs)
    #AX = Y
    return Y,A,X                                

class Lista:
    def __init__(self, n_features, n_components, initial_lambda=1e-1, T=6, learning_rate=1e-3, activation=soft_threshold, model_path="./models"):
        tf.reset_default_graph()
        self.N = n_features
        self.M = n_components
        self._lambda = initial_lambda
        self.T = T
        self.learning_rate = learning_rate
        self.activation = activation
        self.model_path = model_path

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='input')
        self.We = tf.random_uniform(shape=(n_components, n_features), dtype=tf.float32, name='We')                                                             
        
        s,_,_ = tf.svd(tf.matmul(self.We, tf.transpose(self.We)))
        # L0 = np.array(tf.reduce_max(s) ** 2).astype(np.float32)
        self.Lipschitz = tf.Variable(tf.reduce_max(s) ** 2, dtype=tf.float32, name='L')        
        self.B = tf.Variable(tf.matmul(self.We, self.x), dtype=tf.float32, name='B')        
        self.S = tf.Variable(tf.eye(self.M) - 1/self.Lipschitz * (tf.matmul(self.We, tf.transpose(self.We))), dtype=tf.float32, name='S')
        self.theta = tf.Variable(self._lambda / 2 / self.Lipschitz, dtype=tf.float32, name='theta')        
        self.Z = tf.Variable(np.zeros(shape=(n_components)),  dtype=tf.float32, name='Z')
        
        print('We: ', self.We.shape)
        print('x: ',  self.x.shape)
        print('L: ',  self.Lipschitz.shape)
        print('B: ',  self.B.shape)
        print('S: ',  self.S.shape)
        print('z: ',  self.Z.shape)

        self.layers = []
        for i in range(T):
            # self.Lipschitz = tf.get_variable(name='L')
            # self.theta = tf.get_variable(name='theta')
            self.Z = activation(self.B + tf.matmul(self.S, self.Z), self.theta)
            self.layers.append(('Lista T='+str(i+1), self.Z, (self.We, self.S, self.theta)))
        
        self.loss = tf.norm.l2_loss(self.Z - self.x) + tf.norm(self.Z, ord='1')
        self.var_list = [self.We, self.S, self.theta]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.var_list)
        self.merged_summary = tf.summary.merge_all()

    def learn(self, x_train, max_iter=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(max_iter):
                sess.run(self.k)
                sess.run(self.merged_summary, feed_dict={self.x: x_train})
                print('Epoch %s/%s','Loss = ',self.loss, ', L = ', self.Lipschitz %(i,max_iter))
def main():
    X,A,Z = get_data(n_features=30, n_components=50)
    # X = AZ
    N,M = A.shape
    # print(A.shape)
    # print(X.shape)
    # print(Z.shape)
    model = Lista(n_features=N, n_components=M)
    model.learn(x_train=X.T, max_iter=20)

if __name__ == '__main__':
    main()
# def gen(N,A M):    
#     n_samples = 10
#     #X = AZ
#     X,A,Z = make_sparse_coded_signal(n_samples=n_samples, n_components=M, n_features=N, n_nonzero_coefs=10, random_state=1)
    
#     x_placeholder = tf.placeholder(dtype=tf.float32, shape=(N, n_samples))
#     z_placeholder = tf.placeholder(dtype=tf.float32, shape=(M, n_samples))
#     data = tf.data.Dataset.from_tensor_slices((x_placeholder, z_placeholder))
#     iterator = data.make_initializable_iterator()

#     return iterator

# def Lista(X, Wd, T, L=1e-1):
#     activation = soft_threshold
#     layers = []
#     n,m = Wd.shape
#     We_ = 1/L * Wd.T
#     We = tf.Variable(We_, dtype=tf.float32, name='We_0')
#     S_ = tf.eye(m) - tf.matmul(We_, Wd)
#     S = tf.Variable(S_, dtype=tf.float32, name='S_0')
#     B = tf.Variable(tf.matmul(We, X))
#     layers.append(('Linear', B, None))

#     L = np.array(L).astype(np.float32)

#     L0 = tf.Variable(L, name='L0')
#     Z = activation(B, L0)
#     layers.append(('Lista T=1',Z, (L, )))
#     for t in range(1,T):
#         L = tf.Variable(L,name='L_{0}'.format(t))
#         Z = activation(tf.matmul(S, Z) + B, L)
#         layers.append(('Lista T='+str(t+1), Z, (We, L)))
#     return layers

# def setup(layer, Wd, x):
#     stages = []
#     for name, z, vlist in layer:
#         loss_ = tf.nn.l2_loss(z - x) + tf.norm(z, ord='1')
#         train_ = tf.train.AdamOptimizer().minimize(var_list=vlist)

#         stages.append((name, z, loss_, train_))

#     return stages

# def fit(stages, Wd, x, maxit = 1e+5):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for name, z, loss_, train_ in stages:
#             print(name, 'for', ','.join([v for v in tf.trainable_variables()]))
#             for i in range(maxit):
#                 if i%1000 == 0:
#                     print("i={i}, loss = {loss} ".format(i=i,loss=loss_))

#             sess.run(train_, feed_dict={})
        
#     return sess

# def main():
#     iterator = gen(30, 50)
#     s
# if __name__ == '__main__':
#     main()