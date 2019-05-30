import numpy as np
import tensorflow as tf
from sklearn.datasets import make_sparse_coded_signal
import scipy.io as sio

def soft_threshold(x, threshold, iter=-1):            
    #sign(x) * max(0, |x| -T)
    return tf.multiply(tf.sign(x), tf.maximum(tf.abs(x) - threshold, 0), name="Z_{}".format(iter))    

def get_data(n_samples=100, n_features=30, n_components=50, n_nonzero_coefs=7, random_state=None):
    Y,A,X = make_sparse_coded_signal(n_samples=n_samples,n_features=n_features,n_components=n_components
                                    ,n_nonzero_coefs=n_nonzero_coefs, random_state=random_state)
    #AX = Y
    if len(X.shape) == 1:
        Y = np.reshape(Y, (Y.shape[0],1))
        X = np.reshape(X, (X.shape[0],1))

    return Y.T,A,X.T                                

class Lista:
    def __init__(self, n_features, n_components, We, initial_theta=1e-1, T=6, learning_rate=1e-3, activation=soft_threshold, model_name="lista", model_path="./models", tb_path="./tb/"):
        """Args
            n_features, n_components: dim of Wd = dim of We.T

            We = Weight matrix to learn (1/L * Wd)

            initial_theta = initial value of the threshold for the activation

            T = number of cells in the layer

            learning_rate = learning rate for ADAM

            activation = the activation function to use
                    
        Creates the Builds the computational graph for LISTA, mse loss, optimizer and sets up Tensorboard
        """
        tf.reset_default_graph()
        self.N = n_features
        self.M = n_components
        self.T = T
        self.learning_rate = learning_rate
        self.activation = activation
        self.model_path = model_path

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='input')
        self.zmin = tf.placeholder(dtype=tf.float32, shape=(None, n_components), name='optimal_output')
        
        self.We = tf.Variable(We,dtype=tf.float32, name='We') #tf.random_uniform(shape=(n_components, n_features)), dtype=tf.float32, name='We')                                                            
        self.B = tf.matmul(self.We, tf.transpose(self.x), name='B')
        self.S = tf.Variable(tf.eye(self.M) - (tf.matmul(self.We, tf.transpose(self.We), name="inside_S_mul")), dtype=tf.float32, name='S')
        self.theta = tf.Variable(initial_theta, dtype=tf.float32, name='theta')
        
        self.Z_list = []

        print('We:',  self.We.shape)
        print('x: ',  self.x.shape)
        print('B: ',  self.B.shape)
        print('S: ',  self.S.shape)

        old_Z = tf.zeros((n_components,1), tf.float32)

        for i in range(T):
            Z = activation(self.B + tf.matmul(self.S, old_Z), self.theta, i)
            self.Z_list.append(Z)
            old_Z = Z
        
        self.Z = Z
    
        # TensorBoard Setup
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(tb_path + model_name)
        

        self.loss = tf.losses.mean_squared_error(labels=tf.transpose(self.zmin),predictions=self.Z)
        tf.summary.scalar('l2_loss: ', self.loss)
    
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.We = tf.nn.l2_normalize(self.We, axis=0)

        self.merged_summary = tf.summary.merge_all()
        with tf.Session() as sess:
            self.writer.add_graph(sess.graph)
                   

    def learn(self, x_train, z_min, max_iter=100):
        """Trains the model on the data
        Args
            x_train: the input matrix (n_samples, n_features)

            z_min: the actual sparse vectors (n_samples, n_components)

            max_iter: the total epochs to train for
        Returns None
        """

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(max_iter):
                loss, _, theta, summary, z = sess.run([self.loss, self.optimizer, self.theta, 
                                                            self.merged_summary, self.Z], 
                                                            feed_dict = {self.x: x_train, self.zmin: z_min}
                                                    )

                self.writer.add_summary(summary, i)
                print("Epoch", i, "/",max_iter, " Loss = ", loss, "norm err: ", np.linalg.norm(z.T-z_min)**2, z.T.shape) #loss, "lipschitz: ", lips, " theta: ", theta)
        
    def predict(self, x_train, z0):
        """x_train: the input (n_samples,n_features)

            z0: the actual sparse vector
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            z = sess.run([self.Z], feed_dict={self.x: x_train,self.zmin: z0})
            z = z[0]

            print(z.shape)
            print(z0.shape)
            print('rel error: ', np.linalg.norm(z.T-z0)/np.linalg.norm(z0))
            print(z.T[0])
            print(z0[0])
def main():
    X,A,Z = get_data(n_samples=100, n_features=30, n_components=50, n_nonzero_coefs=7)
    print('Got data')
    N,M = A.shape
    # A = np.random.normal(size=(N,M))
    
    e,_ = np.linalg.eigh(A.T.dot(A))
    l = np.max(e)

    model = Lista(n_features=N, n_components=M,We=A.T*(1/l), initial_theta=0.1/2/l, T=100)
    model.learn(x_train=X, z_min= Z, max_iter=7)
    model.predict(x_train=X[1,None], z0=Z[1,None])

if __name__ == '__main__':
    main()