import tensorflow as tf
import numpy as np
import compute_grad as cgrad
import matplotlib.pyplot as plt
rng = np.random
NIN=3
NOUT=2
bs=5
z = np.asarray(np.random.randn(bs,NIN) + 1*np.random.randn(bs,NIN), dtype=np.float32)
z_tf = tf.constant(z)
# z_tf_mat=tf.expand_dims(z_tf,0)
z_tf_mat=z_tf
z_tf_lst=tf.unpack(z_tf)
# f = z_tf**2

# Set model weights
W1 = tf.Variable(rng.randn(NIN,1024), name="weight",dtype=tf.float32)
b1 = tf.Variable(rng.randn(1024), name="bias",dtype=tf.float32)

# Set model weights
W2 = tf.Variable(rng.randn(1024,NOUT), name="weight",dtype=tf.float32)
b2 = tf.Variable(rng.randn(NOUT), name="bias",dtype=tf.float32)



# Set model weights
W3 = tf.Variable(rng.randn(NOUT,NOUT), name="weight",dtype=tf.float32)
b3 = tf.Variable(rng.randn(NOUT), name="bias",dtype=tf.float32)

mid=tf.add(tf.matmul(tf.add(tf.matmul(z_tf_mat**2, W1), b1),W2),b2)
f = tf.squeeze(mid)
f_lst = tf.unpack(mid)
final=tf.add(tf.matmul(mid, W3), b3)

P = tf.placeholder(dtype=tf.float32, shape=[NIN, NIN])
A = tf.placeholder(dtype=tf.float32, shape=[NOUT, NIN])

_P =tf.matmul(A,tf.matmul(P,tf.matrix_transpose(A)))

grads=tf.gradients(f, z_tf)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    P_inp= np.asarray(np.diag([1]*NIN),dtype=np.float32)
    lst_anl=[]
    for i in range(bs):
        analytical = cgrad.compute_gradient(z_tf_lst[i], ([NIN]), f_lst[i], ([NOUT]), x_init_value=z)
        lst_anl.append(analytical)

    feed = {P: P_inp,A:analytical}


    df_dz,P_out = sess.run([grads,_P],feed)


print df_dz
print analytical
# print numerical