"""Module implementing simple mine model."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

# from .neural_networks import neural_info_measure
# from .model import SavableModel

class MINE_Model():
    """MINE model.

        input_shape: a list specifying the dimensions of data.
        arch       : a list containg the dimensions of each layer in MINE
        inputs     : a list containing x and y

    """

    def __init__(self, input_shape, arch,numout=3,rulearch = None):
        """Initialize MINE model."""
        self.input_shape = input_shape;
        self.arch = arch;
        self.numout = numout
        self.rulearch = rulearch;
        self.inf_name = "T_theta"
        self.inf_name_2 = "T_theta2"

    def neural_inf_measure(self,inputs,scope,reu=False,actfun=None):

        with tf.variable_scope(scope,reuse=reu) as vs:
            num_layers = len(self.arch);
            x=inputs;
            for i in range(num_layers):
                x = layers.fully_connected(x,self.arch[i],activation_fn=tf.nn.relu);
            output = layers.fully_connected(x,1,activation_fn=actfun);

        # self.variables = tf.contrib.framework.get_variables(vs)
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs)

        return output #, self.variables

# Need a function like this for saving. Should really use the abc you already have, but its a bit harder to read if
# someone else wants to look at it. But also necessary for sacred experiments as you have everything wrapped now.
    def variables(self):
        return tf.trainable_variables(scope=self.inf_name)

    def variableslambda(self):
        return tf.trainable_variables(scope="lambda") + tf.trainable_variables(scope=self.inf_name_2)


    def loss(self, inputs):
        """Negative loss as using gradient descent. The dimension of inputs must be [batch_axis,...]."""
        x_in, y_in = inputs;

        # shuffle and concatenate
        y_shuffle = tf.random_shuffle(y_in);
        xy = tf.concat([x_in,y_in],axis=1);
        x_y = tf.concat([x_in,y_shuffle],axis=1);

        T_xy = self.neural_inf_measure(xy,self.inf_name);
        T_x_y = self.neural_inf_measure(x_y,self.inf_name,reu=True);

        # compute the negative objective (maximise objective == minimise -objective)
        neg_obj = -(tf.reduce_mean(T_xy) - tf.log(tf.reduce_mean(tf.exp(T_x_y))));

        return neg_obj
