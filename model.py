import tensorflow as tf

class MLP(tf.keras.layers.Layer):
    def __init__(self, filters, layers = 1, activation = None, expand = 2, **kwargs):
        super().__init__(**kwargs)
        self.forward_layers = []
        for _ in range(layers - 1):
            self.forward_layers.append(tf.keras.layers.Dense(filters * expand, kernel_initializer = 'lecun_normal', activation = 'selu'))
        self.forward_layers.append(tf.keras.layers.Dense(filters, kernel_initializer = 'lecun_normal', activation = activation))
    
    def call(self, x, training, **kwargs):
        for l in self.forward_layers:
            x = l(x, training = training)
        return x

class add_noise(tf.keras.layers.Layer):
    def __init__(self, noise_degree = .1):
        super().__init__(name = 'add_noise')
        self.noise_degree = noise_degree
        
    def call(self, x, training = True, **kwargs):
        if training:
            shape = tf.shape(x)
            thr = tf.random.uniform([shape[0]])
            for i in range(len(shape) - 1):
                thr = thr[:, None]
            rmap = tf.nn.relu(tf.math.sign(tf.random.uniform(shape) - thr))
            rand = tf.random.normal(shape) * self.noise_degree
            return rmap * rand + x
        else:
            return x

class MODEL(tf.keras.Model):
    def __init__(self, inputs, outputs, latents, median, learning_rate, init_beta, clip_KL, stair_height, stair_gap, batchsize, with_noise):
        super(MODEL, self).__init__(name = 'model')
        
        self.add_noise = add_noise(.05)
        self.batchsize = batchsize
        self.miu_z_mean = tf.zeros(shape = [latents,])
        self.miu_z_std = tf.zeros(shape = [latents,])
        self.inputs = inputs
        self.outputs = outputs
        self.latents = latents
        self.init_beta = init_beta
        self.clip_KL = clip_KL
        self.stair_gap = stair_gap
        self.stair_height = stair_height
        self.with_noise = with_noise

        # 定义利用 O 生成 E^ 的 ENDR1
        self.ENDR_main = MLP(outputs, layers = 3, name = 'ENDR_main')
        # 定义利用 O 生成 Z 的 ENDR2
        self.ENDR_prog = MLP(latents * 2, layers = 3, name = 'ENDR_prog')
        # 定义利用 E 生成 Median 的 DECR1
        self.DECR_main = MLP(median, layers = 3, name = 'DECR_main')
        # 定义利用 Z 生成 Patch 的 DECR2
        self.DECR_prog = MLP(median, layers = 3, name = 'DECR_prog')
        # 定义利用 Median 及 Patch 过的 Median 重构 O 的 DECR3
        self.DECR_fina = MLP(inputs, layers = 3, name = 'DECR_fina')
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps = 100000,
            decay_rate = 0.7071,
            staircase = True)
        self.op = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule,
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-07,
            amsgrad = False)
        
        self.trained_iterations = 0

    def Encode_main(self, x):
        miu_y = self.ENDR_main(x)
        return miu_y
    
    def Encode_prog(self, x):
        z_ = self.ENDR_prog(x)
        miu_z = z_[:, :self.latents]
        sig_log_z = z_[:, self.latents:]
        return miu_z, -tf.nn.elu(-sig_log_z)
    
    def Decode_main(self, y):
        median = self.DECR_main(y[:, :self.outputs])
        return median
    
    def Decode_prog(self, z):
        patch = self.DECR_prog(z)
        return patch
    
    def Decode_fina(self, median, patch):
        miu_x = self.DECR_fina(median + patch)
        return miu_x
        
    def train(self, x, y):
        if self.with_noise:
            miu_y = self.Encode_main(self.add_noise(x))
            miu_z, sig_log_z = self.Encode_prog(self.add_noise(x))
        else:
            miu_y = self.Encode_main(x)
            miu_z, sig_log_z = self.Encode_prog(x)
        
        if len(miu_z) >= self.batchsize:
            self.KL = self.KL_loss(miu_z, sig_log_z)
            self.KL_threshold = tf.nn.relu(tf.math.sign(self.KL - 1))
            self.miu_z_mean = tf.math.reduce_mean(miu_z, axis = 0)
            self.miu_z_std = tf.math.reduce_std(miu_z, axis = 0)
            
        z = miu_z + tf.random.normal(tf.shape(miu_z)) * tf.math.exp(sig_log_z)
        median = self.Decode_main(y[:, :self.outputs])

        if tf.reduce_max(self.KL) >= self.clip_KL:
            patch = self.Decode_prog(z * self.KL_threshold)
        else:
            patch = self.Decode_prog(z)
        
        miu_x_0 = self.Decode_fina(median, 0)
        miu_x_1 = self.Decode_fina(median, patch)
        return miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1

    def call(self, x, y):
        miu_y = self.Encode_main(x)        
        miu_z, sig_log_z = self.Encode_prog(x)
        try:
            z = miu_z * self.KL_threshold
        except:
            self.KL = self.KL_loss(miu_z, sig_log_z)
            self.KL_threshold = tf.nn.relu(tf.math.sign(self.KL - 1))
            z = miu_z * self.KL_threshold
            if len(miu_z) < self.batchsize:
                del self.KL_threshold
                print('Require more data (>=%d) to calculate mean/std of miu_z, miu_z_mean/std remain unchanged.'%(self.batchsize))
            else:
                self.miu_z_mean = tf.math.reduce_mean(miu_z, axis = 0)
                self.miu_z_std = tf.math.reduce_std(miu_z, axis = 0)
                print('Recalculated KL_threshold, some training is advised.')    

        median = self.Decode_main(y)
        patch = self.Decode_prog(z)
        miu_x_0 = self.Decode_fina(median, 0)
        miu_x_1 = self.Decode_fina(median, patch)
        return miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1
    
    def predict(self, x, y):
        sumlength = 1
        size = self.batchsize
        miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1 = self.call(x[0: 1], y[0: 1])
        while sumlength <= len(x):
            miu_y_n, miu_z_n, sig_log_z_n, median_n, patch_n, miu_x_0_n, miu_x_1_n = self.call(x[sumlength: sumlength + size], y[sumlength: sumlength + size])
            sumlength += size
            miu_y = tf.concat((miu_y, miu_y_n), axis = 0)
            miu_z = tf.concat((miu_z, miu_z_n), axis = 0)
            sig_log_z = tf.concat((sig_log_z, sig_log_z_n), axis = 0)
            median = tf.concat((median, median_n), axis = 0)
            patch = tf.concat((patch, patch_n), axis = 0)
            miu_x_0 = tf.concat((miu_x_0, miu_x_0_n), axis = 0)
            miu_x_1 = tf.concat((miu_x_1, miu_x_1_n), axis = 0)
        return miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1
    
    def get_miu_z_n(self, miu_z):
        return (miu_z - self.miu_z_mean)/(self.miu_z_std + 1e-8)
    
    def generate(self, y, z_n):
        # only normalized z can be input here!
        z = z_n * self.miu_z_std + self.miu_z_mean
        median = self.Decode_main(y)
        patch = self.Decode_prog(z * self.KL_threshold)
        miu_x_1 = self.Decode_fina(median, patch)
        return miu_x_1
    
    def validation(self, x_train, x_test, y_train, y_test):
        miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1 = self.call(x_train, y_train)
        EM_train_0 = self.PSNR(x_train, miu_x_0)
        EM_train = self.PSNR(x_train, miu_x_1)
        miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1 = self.call(x_test, y_test)
        EM_test = self.PSNR(x_test, miu_x_1)
        z_KL_loss = self.KL_loss(miu_z, sig_log_z)
        return EM_train_0, EM_train, EM_test, z_KL_loss
    
    def PSNR(self, x, x_):
        return 10*tf.math.log(1/self.MSE_loss(x, x_))/tf.math.log(10.)
    
    def MSE_loss(self, y, miu_y):
        return tf.reduce_mean((y - miu_y)**2, axis = 0, keepdims = False)
    
    def KL_loss(self, miu, sig_log):
        ELBO = tf.math.exp(2*sig_log) + miu**2 - 2*sig_log - 1
        return 0.5*tf.reduce_mean(ELBO, axis = 0, keepdims = False)
    
    def losses(self, x, y):
        miu_y, miu_z, sig_log_z, median, patch, miu_x_0, miu_x_1 = self.train(x, y)
        y_recon_loss = tf.reduce_mean(self.MSE_loss(y[:,:self.outputs], miu_y))
        z_KL_loss = tf.reduce_sum(self.KL_loss(miu_z, sig_log_z))
        x_recon_0_loss = tf.reduce_mean(self.MSE_loss(x, miu_x_0))
        x_recon_1_loss = tf.reduce_mean(self.MSE_loss(x, miu_x_1))
        return y_recon_loss, x_recon_0_loss, x_recon_1_loss, z_KL_loss

    def trainNN(self, x, y):
        with tf.GradientTape() as G:
            self.C = self.stair_height**(self.init_beta + self.trained_iterations//self.stair_gap)
            y_recon_loss, x_recon_0_loss, x_recon_1_loss, z_KL_loss = self.losses(x, y)
            loss = y_recon_loss + x_recon_0_loss + x_recon_1_loss + self.C * z_KL_loss
        grads = G.gradient(loss, self.trainable_variables)
        self.op.apply_gradients(grads_and_vars = zip(grads, self.trainable_variables))
        self.trained_iterations += 1
        return grads