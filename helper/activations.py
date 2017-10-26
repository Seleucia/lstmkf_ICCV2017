import tensorflow as tf

HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)
lin_sigmoid = lambda x: 0.25 * x + 0.5


# Sigmoid = lambda x, use_noise=0: T.nnet.sigmoid(x)
HardSigmoid = lambda x, angle=0.25: tf.maximum(tf.minimum(angle*x + 0.5, 1.0), 0.0)
HardSigmoid = lambda x: tf.minimum(tf.maximum(lin_sigmoid(x), 0.), 1.)



#https://github.com/caglar/noisy_units/blob/master/codes/theano/nunits.py
#Tensorflow port for Noisy Activation Functions paper.
def NHardTanh(x,
              use_noise,
              c=0.05):
    """
    Noisy Hard Tanh Units: NANI as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    threshold = 1.001

    def noise_func() :return tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def zero_func (): return tf.zeros(tf.shape(x), dtype=tf.float32, name=None)

    noise=tf.cond(use_noise,noise_func,zero_func)

    res = HardTanh(x + c * noise)
    return res

def NHardSigmoid(x,
                 use_noise,
                 c=0.05):
    """
    Noisy Hard Sigmoid Units: NANI as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """

    def noise_func() :return tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def zero_func (): return tf.zeros(tf.shape(x), dtype=tf.float32, name=None)

    noise=tf.cond(use_noise,noise_func,zero_func)

    res = HardSigmoid(x + c * noise)
    return res

def NHardTanhSat(x,
                 use_noise,
                 c=0.25):
    """
    Noisy Hard Tanh Units at Saturation: NANIS as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    threshold = 1.001
    def noise_func() :return tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def zero_func (): return tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    noise=tf.cond(use_noise,noise_func,zero_func)


    test = tf.cast(tf.greater(tf.abs(x) , threshold), tf.float32)
    res = test * HardTanh(x + c * noise) + (1. - test) * HardTanh(x)
    return res

def NHardSigmoidSat(x,
                    use_noise,
                    c=0.25):
    """
    Noisy Hard Sigmoid Units at Saturation: NANIS as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    threshold = 1.001
    def noise_func() :return tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def zero_func (): return tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    noise=tf.cond(use_noise,noise_func,zero_func)


    test = tf.cast(tf.greater(tf.abs(x) , threshold), tf.float32)
    res = test * HardSigmoid(x + c * noise) + \
            (1. - test) + HardSigmoid(x)
    return res

def NTanh(x,
          use_noise,
          alpha=1.05,
          c=0.5, half_normal=False):
    """
    Noisy Hard Tanh Units: NAN without learning p
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: the leaking rate from the linearized function to the nonlinear one.
    """


    threshold = 1.0
    signs = tf.sign(x)
    delta = tf.abs(x) - threshold

    scale = c * (tf.sigmoid(delta**2) - 0.5)**2
    if alpha > 1.0 and  half_normal:
           scale *= -1.0
    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    def noise_func() :return tf.abs(tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32))
    def zero_func (): return zeros+ 0.797  if half_normal   else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)

    eps = scale * noise + alpha * delta
    z = x - signs * eps
    test=tf.cast(tf.greater_equal(tf.abs(x) , threshold),tf.float32)
    res = test * z + (1. - test) *  HardTanh(x)


    return res


def NSigmoid(x,
              use_noise,
              alpha=1.15,
              c=0.25, half_normal=False):
    """
    Noisy Hard Sigmoid Units: NAN without learning p
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: the leaking rate from the linearized function to the nonlinear one.
    """

    threshold=2.0,
    delta = tf.abs(x) - threshold

    scale = c * (tf.sigmoid(delta**2) - 0.5)**2

    if alpha > 1.0 and  half_normal:
           scale *= -1.0
    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    def noise_func() :return tf.abs(tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32))
    def zero_func (): return zeros+ 0.797  if half_normal   else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)


    eps = scale * noise + alpha * delta
    signs = tf.sign(x)
    z = x - signs * eps

    test = tf.cast(tf.greater_equal(tf.abs(x) , threshold),tf.float32)
    res = test * z + (1. - test) * HardSigmoid(x)

    return res


def NTanhP(x,
           p,
           use_noise,
           alpha=1.15,
           c=0.5,
           noise=None,
           clip_output=False,
           half_normal=False):
    """
    Noisy Hard Tanh Units: NAN with learning p
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        p: tensorflow variable, a vector of parameters for p.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: float, the leakage rate from the linearized function to the nonlinear one.
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """


    if not noise:
        noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)

    signs = tf.sign(x)
    delta = HardTanh(x) - x

    scale = c * (tf.sigmoid(p * delta) - 0.5)**2
    if alpha > 1.0 and  half_normal:
           scale *= -1.0


    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    rn_noise=tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def noise_func() :return tf.abs(rn_noise) if half_normal   else zeros
    def zero_func (): return zeros+ 0.797  if half_normal   else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)

    res = alpha * HardTanh(x) + (1. - alpha) * x - signs * scale * noise

    if clip_output:
        return HardTanh(res)
    return res


def NSigmoidP(x,
              p,
              use_noise,
              alpha=1.1,
              c=0.15,
              noise=None,
              half_normal=True):
    """
    Noisy Sigmoid Tanh Units: NAN with learning p
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        p: tensorflow shared variable, a vector of parameters for p.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: float, the leakage rate from the linearized function to the nonlinear one.
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """
    lin_sigm = 0.25 * x + 0.5
    signs = tf.sign(x)
    delta = HardSigmoid(x) - lin_sigm
    signs = tf.sign(x)
    scale = c * (tf.sigmoid(p * delta) - 0.5)**2
    if not noise:
        noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)

    if alpha > 1.0 and  half_normal:
           scale *= -1.0


    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    rn_noise=tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def noise_func() :return tf.abs(rn_noise) if half_normal   else rn_noise
    def zero_func (): return zeros+ 0.797  if half_normal   else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)
    res = (alpha * HardSigmoid(x) + (1. - alpha) * lin_sigm - signs * scale * noise)

    return res

def NSigmoidPInp(x,
               p,
               use_noise,
               alpha=1.1,
               c=0.25,
               half_normal=False):
    """
    Noisy Sigmoid where the noise is injected to the input: NANI with learning p.
    This function works well with discrete switching functions.
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        p: tensorflow shared variable, a vector of parameters for p.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """

    signs = tf.sign(x)
    delta = HardSigmoid(x) - (0.25 * x + 0.5)
    signs = tf.sign(x)
    noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)


    if half_normal and alpha > 1.0:
          c *= -1.0

    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    rn_noise=tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def noise_func() :return tf.abs(rn_noise) if half_normal   else rn_noise
    def zero_func (): return zeros+ 0.797  if half_normal  else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)
    scale = c * tf.nn.softplus(p * abs(delta) / (abs(noise) + 1e-10))
    res = HardSigmoid(x + scale * noise)

    return res

def NTanhPInp(x,
              p,
              use_noise,
              alpha=1.1,
              c=0.25,
              half_normal=False):
    """
    Noisy Tanh units where the noise is injected to the input: NANI with learning p.
    This function works well with discrete switching functions.
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        p: tensorflow shared variable, a vector of parameters for p.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """

    signs = tf.sign(x)
    delta = HardTanh(x) - x
    signs =  tf.sign(x)
    noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)

    noise_det = 0.
    if half_normal and  alpha > 1.0:
          c *= -1

    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    rn_noise=tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    def noise_func() :return tf.abs(rn_noise) if half_normal   else rn_noise
    def zero_func (): return zeros+ 0.797  if half_normal  else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)

    scale = c * tf.nn.softplus(p * abs(delta) / (abs(noise) + 1e-10))
    res = HardTanh(x + scale * noise)
    return res