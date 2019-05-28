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
        self.We = tf.Variable(np.random.uniform(size=(n_components, n_features)), dtype=tf.float32, name='We')                                                            
        
        s,_,_ = tf.svd(tf.matmul(self.We, tf.transpose(self.We)))
        self.Lipschitz = tf.reduce_max(s) ** 2        

        self.B = tf.matmul(self.We, tf.transpose(self.x))        
        self.S = tf.eye(self.M) - 1/self.Lipschitz * (tf.matmul(self.We, tf.transpose(self.We)))
        self.theta = tf.Variable(self._lambda / 2 / self.Lipschitz, dtype=tf.float32, name='theta')        
        self.Z = tf.Variable(np.zeros(shape=(n_components, 1)),  dtype=tf.float32, name='Z')
        
        print('We:', self.We.shape)
        print('x: ',  self.x.shape)
        print('L: ',  self.Lipschitz.shape)
        print('B: ',  self.B.shape)
        print('S: ',  self.S.shape)
        print('z: ',  self.Z.shape)

        self.layers = []
        for i in range(T):
            self.Z = activation(self.B + tf.matmul(self.S, self.Z), self.theta)
            self.layers.append(('Lista T='+str(i+1), self.Z, (self.We, self.theta)))
        
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.Z - tf.matmul(self.We, tf.transpose(self.x))) + tf.norm(self.Z, ord=1))
        print('loss: ', self.loss.shape)
        tf.summary.scalar('loss: ', self.loss)
        
        self.var_list = [self.We,self.theta]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.var_list)
        self.merged_summary = tf.summary.merge_all()

        print(self.layers)

    def learn(self, x_train, max_iter=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(max_iter):
                sess.run(self.optimizer, feed_dict={self.x: x_train})
                print("Epoch", i, "/",max_iter, " Loss = ",self.loss)
def main():
    X,A,Z = get_data(n_features=30, n_components=50)
    # X = AZ
    N,M = A.shape
    
    model = Lista(n_features=N, n_components=M)
    model.learn(x_train=X.T, max_iter=20)

if __name__ == '__main__':
    main()