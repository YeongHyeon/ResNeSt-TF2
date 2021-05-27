import tensorflow as tf

class Layers(object):

    def __init__(self):

        self.parameters = {}
        self.num_params = 0
        self.initializer_xavier = tf.initializers.glorot_normal()

    def elu(self, inputs): return tf.nn.elu(inputs)
    def relu(self, inputs): return tf.nn.relu(inputs)
    def sigmoid(self, inputs): return tf.nn.sigmoid(inputs)
    def softmax(self, inputs): return tf.nn.softmax(inputs)

    def dropout(self, inputs, rate): return tf.nn.dropout(inputs, rate=rate)

    def maxpool(self, inputs, pool_size, stride_size):

        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], \
            padding='VALID', strides=[1, stride_size, stride_size, 1])

    def avgpool(self, inputs, pool_size, stride_size):

        return tf.nn.avg_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], \
            padding='VALID', strides=[1, stride_size, stride_size, 1])

    def get_weight(self, vshape, transpose=False, bias=True, name=""):

        try:
            w = self.parameters["%s_w" %(name)]
            b = self.parameters["%s_b" %(name)]
        except:
            w = tf.Variable(self.initializer_xavier(vshape), \
                name="%s_w" %(name), trainable=True, dtype=tf.float32)
            self.parameters["%s_w" %(name)] = w

            tmpparams = 1
            for d in vshape: tmpparams *= d
            self.num_params += tmpparams

            if(bias):
                if(transpose): b = tf.Variable(self.initializer_xavier([vshape[-2]]), \
                    name="%s_b" %(name), trainable=True, dtype=tf.float32)
                else: b = tf.Variable(self.initializer_xavier([vshape[-1]]), \
                    name="%s_b" %(name), trainable=True, dtype=tf.float32)
                self.parameters["%s_b" %(name)] = b

                self.num_params += vshape[-2]

        if(bias): return w, b
        else: return w

    def fullcon(self, inputs, variables):

        [weights, biasis] = variables
        out = tf.matmul(inputs, weights) + biasis

        return out

    def conv2d(self, inputs, variables, stride_size, padding):

        [weights, biasis] = variables
        out = tf.nn.conv2d(inputs, weights, \
            strides=[1, stride_size, stride_size, 1], padding=padding) + biasis

        return out

    def batch_norm(self, inputs, name=""):

        # https://arxiv.org/pdf/1502.03167.pdf

        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        var = std**2

        try:
            offset = self.parameters["%s_offset" %(name)]
            scale = self.parameters["%s_scale" %(name)]
        except:
            offset = tf.Variable(0, \
                name="%s_offset" %(name), trainable=True, dtype=tf.float32)
            self.parameters["%s_offset" %(name)] = offset
            self.num_params += 1
            scale = tf.Variable(1, \
                name="%s_scale" %(name), trainable=True, dtype=tf.float32)
            self.parameters["%s_scale" %(name)] = scale
            self.num_params += 1

        offset # zero
        scale # one
        out = tf.nn.batch_normalization(
            x = inputs,
            mean=mean,
            variance=var,
            offset=offset,
            scale=scale,
            variance_epsilon=1e-12,
            name=name
        )

        return out
