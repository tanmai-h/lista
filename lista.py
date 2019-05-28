import numpy as np
import tensorflow as tf
from sklearn.datasets import make_sparse_coded_signal
import scipy.io as sio

def soft_threshold(x, threshold, iter=-1):            
    #sign(x) * max(0, |x| -T)
    return tf.multiply(tf.sign(x), tf.maximum(tf.abs(x) - threshold, 0), name="Z_{}".format(iter))    

def get_data(n_samples=100, n_features=30, n_components=50, n_nonzero_coefs=7):
    Y,A,X = make_sparse_coded_signal(n_samples=n_samples,n_features=n_features,n_components=n_components
                                    ,n_nonzero_coefs=n_nonzero_coefs)
    #AX = Y
    return Y,A,X                                

class Lista:
    def __init__(self, n_features, n_components, initial_lambda=1e-1, T=6, learning_rate=1e-3, activation=soft_threshold, model_name="lista", model_path="./models", tb_path="./tb/"):
        tf.reset_default_graph()
        self.N = n_features
        self.M = n_components
        self._lambda = initial_lambda
        self.T = T
        self.learning_rate = learning_rate
        self.activation = activation
        self.model_path = model_path

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='input')
        rand_mat = np.random.uniform(size=(n_components, n_features))
        print("AAAAAAAAAAAAAAAA rand_mat = ", rand_mat)
        self.We = tf.Variable(rand_mat, dtype=tf.float32, name='We')                                                            
        
        s,_,_ = tf.svd(tf.matmul(self.We, tf.transpose(self.We)), name='svd_op')
        self.Lipschitz = tf.reduce_max(s) ** 2

        self.B = tf.matmul(self.We, tf.transpose(self.x), name='B')        
        self.S = tf.eye(self.M) - 1/self.Lipschitz * (tf.matmul(self.We, tf.transpose(self.We), name="inside_S"))
        self.theta = self._lambda / 2 / self.Lipschitz
        #self.Z = tf.Variable(np.zeros(shape=(n_components, 1)),  dtype=tf.float32, name='Z')
        self.Z_list = []

        print('We:',  self.We.shape)
        print('x: ',  self.x.shape)
        print('L: ',  self.Lipschitz.shape)
        print('B: ',  self.B.shape)
        print('S: ',  self.S.shape)

        old_Z = tf.zeros((n_components, 1), tf.float32)

        for i in range(T):
            Z = activation(self.B + tf.matmul(self.S, old_Z), self.theta, i)
            self.Z_list.append(Z)
            old_Z = Z
        
        self.Z = Z

        print('Z_list: ',  len(self.Z_list))

        # TensorBoard Setup
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(tb_path + model_name)
        

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.Z - tf.matmul(self.We, tf.transpose(self.x))) + tf.norm(self.Z, ord=1))
        tf.summary.scalar('f_loss: ', self.loss)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        self.merged_summary = tf.summary.merge_all()
        with tf.Session() as sess:
            self.writer.add_graph(sess.graph)
        

    def learn(self, x_train, max_iter=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(max_iter):
                loss, _, B, S, theta, lips = sess.run([self.loss, self.optimizer, self.We, self.S, self.theta, self.Lipschitz], feed_dict = {self.x: x_train})
                
                #summary = sess.run([self.merged_summary], feed_dict={self.x: x_train})
                print("Epoch", i, "/",max_iter, " Loss = ", loss)
                print("lipschitz = ", lips, "We = ", B, "S = ", S, "theta = ", theta)
def main():
    X,A,Z = get_data(n_features=30, n_components=50)
    # X = AZ
    N,M = A.shape
    
    model = Lista(n_features=N, n_components=M)
    model.learn(x_train=X.T, max_iter=20)

if __name__ == '__main__':
    main()