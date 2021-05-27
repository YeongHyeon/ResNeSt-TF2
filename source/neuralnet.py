import os
import tensorflow as tf
import source.layers as lay

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

class Agent(object):

    def __init__(self, height, width, channel, num_class, \
        ksize, radix=4, kpaths=4, learning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-ResNeSt...")
        self.height, self.width, self.channel, self.num_class, self.ksize, self.radix, self.kpaths = \
            height, width, channel, num_class, ksize, radix, kpaths
        self.learning_rate = learning_rate
        self.ckpt_dir = ckpt_dir

        self.__model = Neuralnet(height, width, channel, num_class, ksize, radix, kpaths)
        self.__model.forward(x=tf.zeros((1, height, width, channel), dtype=tf.float32), verbose=True)

        self.variables = {}
        self.__init_propagation(path=self.ckpt_dir)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(path)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

        conc_func = self.__model.__call__.get_concrete_function(tf.TensorSpec(shape=(1, self.height, self.width, self.channel), dtype=tf.float32))
        self.__get_flops(conc_func)

    @tf.autograph.experimental.do_not_convert
    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            logits = self.__model.forward(x, verbose=False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = self.__model.layer.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('ResNeSt/loss', loss, step=iteration)
                tf.summary.scalar('ResNeSt/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def __get_flops(self, conc_func):

        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(conc_func)

        with tf.Graph().as_default() as graph:
            tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

            flop_tot = flops.total_float_ops
            ftxt = open("flops.txt", "w")
            for idx, name in enumerate(['', 'K', 'M', 'G', 'T']):
                text = '%.3f [%sFLOPS]' %(flop_tot/10**(3*idx), name)
                print(text)
                ftxt.write("%s\n" %(text))
            ftxt.close()


    def save_params(self, model='base', tflite=False):

        if(tflite):
            # https://github.com/tensorflow/tensorflow/issues/42818
            conc_func = self.__model.__call__.get_concrete_function(tf.TensorSpec(shape=(1, self.height, self.width, self.channel), dtype=tf.float32))
            converter = tf.lite.TFLiteConverter.from_concrete_functions([conc_func])

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

            tflite_model = converter.convert()

            with open('model.tflite', 'wb') as f:
                f.write(tflite_model)
        else:
            vars_to_save = self.__model.layer.parameters.copy()
            vars_to_save["optimizer"] = self.optimizer

            ckpt = tf.train.Checkpoint(**vars_to_save)
            ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.ckpt_dir, model), max_to_keep=1)
            ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.__model.layer.name_bank):
            vars_to_load[self.__model.layer.name_bank[idx]] = self.__model.layer.parameters[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

class Neuralnet(tf.Module):

    def __init__(self, height, width, channel, num_class, \
        ksize, radix=4, kpaths=4):
        super(Neuralnet, self).__init__()

        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.radix, self.kpaths = ksize, radix, kpaths
        self.layer = lay.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        if(verbose): print("input", x.shape)

        conv1 = self.layer.conv2d(x, \
            self.layer.get_weight(vshape=[3, 3, self.channel, 16], name="%s" %("conv1")), \
            stride_size=1, padding='SAME')
        conv1_bn = self.layer.batch_norm(conv1, name="%s_bn" %("conv1"))
        conv1_act = self.layer.elu(conv1_bn)
        conv1_pool = self.layer.maxpool(conv1_act, pool_size=2, stride_size=2)

        conv2_1 = self.residual_S(conv1_pool, \
            ksize=self.ksize, inchannel=16, outchannel=32, \
            radix=self.radix, kpaths=self.kpaths, name="conv2_1", verbose=verbose)
        conv2_2 = self.residual_S(conv2_1, \
            ksize=self.ksize, inchannel=32, outchannel=32, \
            radix=self.radix, kpaths=self.kpaths, name="conv2_2", verbose=verbose)
        conv2_pool = self.layer.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.residual_S(conv2_pool, \
            ksize=self.ksize, inchannel=32, outchannel=64, \
            radix=self.radix, kpaths=self.kpaths, name="conv3_1", verbose=verbose)
        conv3_2 = self.residual_S(conv3_1, \
            ksize=self.ksize, inchannel=64, outchannel=64, \
            radix=self.radix, kpaths=self.kpaths, name="conv3_2", verbose=verbose)

        [n, h, w, c] = conv3_2.shape
        flat = tf.compat.v1.reshape(conv3_2, shape=[-1, h*w*c], name="flat")
        if(verbose):
            num_param_fe = self.layer.num_params
            print("flat", flat.shape)

        fc1 = self.layer.fullcon(flat, \
            self.layer.get_weight(vshape=[h*w*c, self.num_class], name="fullcon1"))
        if(verbose): print("\nNum Parameter: ", self.layer.num_params)
        return fc1

    def residual_S(self, input, ksize, inchannel, outchannel, \
        radix, kpaths, name="", verbose=False):

        convtmp_1 = self.layer.conv2d(input, \
            self.layer.get_weight(vshape=[ksize, ksize, inchannel, outchannel], name="%s_1" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_1bn = self.layer.batch_norm(convtmp_1, name="%s_1bn" %(name))
        convtmp_1act = self.layer.elu(convtmp_1bn)
        convtmp_2 = self.layer.conv2d(convtmp_1act, \
            self.layer.get_weight(vshape=[ksize, ksize, outchannel, outchannel], name="%s_2" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_2bn = self.layer.batch_norm(convtmp_2, name="%s_2bn" %(name))
        convtmp_2act = self.layer.elu(convtmp_2bn)

        concats_1 = None
        for idx_k in range(kpaths):
            cardinal = self.cardinal(convtmp_2act, ksize, outchannel, outchannel//2, radix, kpaths, name="%s_car_k%d" %(name, idx_k))
            if(idx_k == 0): concats_1 = cardinal
            else: concats_1 = tf.concat([concats_1, cardinal], axis=3)
        concats_2 = self.layer.conv2d(concats_1, \
            self.layer.get_weight(vshape=[1, 1, outchannel//2, outchannel], name="%s_cc" %(name)), \
            stride_size=1, padding='SAME')
        concats_2 = concats_2 + convtmp_2act

        if(input.shape[-1] != concats_2.shape[-1]):
            convtmp_sc = self.layer.conv2d(input, \
                self.layer.get_weight(vshape=[1, 1, inchannel, outchannel], name="%s_sc" %(name)), \
                stride_size=1, padding='SAME')
            convtmp_scbn = self.layer.batch_norm(convtmp_sc, name="%s_scbn" %(name))
            convtmp_scact = self.layer.elu(convtmp_scbn)
            input = convtmp_scact

        output = input + concats_2

        if(verbose): print(name, output.shape)
        return output

    def cardinal(self, input, ksize, inchannel, outchannel, \
        radix, kpaths, name="", verbose=False):

        if(verbose): print("cardinal")
        outchannel_cv11 = int(outchannel / radix / kpaths)
        outchannel_cvkk = int(outchannel / kpaths)

        inputs = []
        for idx_r in range(radix):
            conv1 = self.layer.conv2d(input, \
                self.layer.get_weight(vshape=[1, 1, inchannel, outchannel_cv11], name="%s1_r%d" %(name, idx_r)), \
                stride_size=1, padding='SAME')
            conv1_bn = self.layer.batch_norm(conv1, name="%s1_bn" %(name))
            conv1_act = self.layer.elu(conv1_bn)

            conv2 = self.layer.conv2d(conv1_act, \
                self.layer.get_weight(vshape=[ksize, ksize, outchannel_cv11, outchannel_cvkk], name="%s2_r%d" %(name, idx_r)), \
                stride_size=1, padding='SAME')
            conv2_bn = self.layer.batch_norm(conv2, name="%s2_bn" %(name))
            conv2_act = self.layer.elu(conv2_bn)
            inputs.append(conv2_act)

        return self.split_attention(inputs, outchannel_cvkk, name="%s_att" %(name))

    def split_attention(self, inputs, inchannel, name="", verbose=False):

        if(verbose): print("split attention")
        radix = len(inputs)
        input_holder = None
        for idx_i, input in enumerate(inputs):
            if(idx_i == 0): input_holder = input
            else: input_holder += input

        ga_pool = tf.math.reduce_mean(input_holder, axis=(1, 2))
        ga_pool = tf.expand_dims(tf.expand_dims(ga_pool, axis=1), axis=1)

        dense1 = self.layer.conv2d(ga_pool, \
            self.layer.get_weight(vshape=[1, 1, inchannel, inchannel//2], name="%s1" %(name)), \
            stride_size=1, padding='SAME')
        dense1_bn = self.layer.batch_norm(dense1, name="%s_bn" %(name))
        dense1_act = self.layer.elu(dense1_bn)

        output_holder = None
        for idx_r in range(radix):
            dense2 = self.layer.conv2d(dense1_act, \
                self.layer.get_weight(vshape=[1, 1, inchannel//2, inchannel], name="%s2_r%d" %(name, idx_r)), \
                stride_size=1, padding='SAME')
            if(radix == 1): r_softmax = self.layer.sigmoid(dense2)
            elif(radix > 1): r_softmax = self.layer.softmax(dense2)

            if(idx_r == 0): output_holder = inputs[idx_r] * r_softmax
            else: output_holder += inputs[idx_r] * r_softmax

        return output_holder
