import tensorflow as tf

class GASF(tf.keras.layers.Layer):
    def __init__(self, post_scale = True):
        """use summation for bijective mapping
        post_scale == True -> output range := [0,1]"""
        super(GASF, self).__init__()
        self.post_scale = post_scale

    def call(self, inputs):
        """input-values must be in range [-1,1] - for bijective mapping in [0,1]"""
        X = inputs #[0,1]
        I = tf.ones(shape=tf.shape(inputs))
        I_X = tf.sqrt(I-tf.pow(X,2 * I)) #[0,1]

        G = tf.matmul(X,X,transpose_a=True) - tf.matmul(I_X,I_X,transpose_a=True) 
        # bijective case: mode == summation, inputs in range [0,1] -> output range := [-1, 1] but we wish for images in range [0,1]
        # -> simply min/max scale with min = -1 and max = 1
        if self.post_scale:
            G = (G+1)/2

        #G = tf.matmul(I_X,X,transpose_a=True) - tf.matmul(X,I_X,transpose_a=True) <- Formel für GADF
        return G