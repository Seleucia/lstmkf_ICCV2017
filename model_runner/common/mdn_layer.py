import tensorflow as tf
import numpy as np
import math

class MDN():
    def __init__(self,n_output,NOUT,KMIX,mode):
        self._NOUT=NOUT
        self._n_output=n_output
        self._KMIX=KMIX
        self._mode=mode

    def euclidean_norm(t):
        squareroot_tensor = tf.pow(t,2)
        euclidean_norm = tf.reduce_sum(squareroot_tensor,reduction_indices=1)
        return tf.sqrt(euclidean_norm)

    def tf_multi_bishop_kernel(self,y, mu, sigma):
        #f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
        # C=tf.matrix_diag(sigma)
        # nominator=tf.pow(2.0*math.pi,np.float32(NDIM)/2.0)*tf.pow(sigma,NDIM)
        # sb=tf.reduce_sum(tf.sub(y,mu),reduction_indices=1)
        # expo=tf.exp(-tf.div(tf.pow(sb,2),2.*tf.pow(sigma,2)))
        # return tf.div(expo, nominator)
        # nominator=np.power(2.0*math.pi,(float(self._n_output)/2.0))
        # nominator=1
        # sigma_ndim=tf.pow(sigma,float(self._n_output))
        # nominator=tf.mul(sigma_ndim,nominator)
        # nominator=tf.div(1.0,nominator)

        expe=tf.abs(tf.sub(y,mu))
        squareroot_tensor = tf.pow(expe,2)
        sqrt_norm = tf.reduce_sum(squareroot_tensor,reduction_indices=1)
        # tens_norm=tf.sqrt(euclidean_norm)
        # expe=tf.pow(tens_norm,2.0)
        # expe=euclidean_norm
        exp_in=tf.div(sqrt_norm,2.0*tf.pow(sigma,2.0))
        # expe=tf.exp(exp_in)

        # return tf.div(expe,nominator)
        return exp_in

    def get_mixture_coef(self,final_output):
      lst=tf.unpack(final_output, axis=1)
      #144=KMIX * (NDIM+3)=24*(3+3)
      # out_pi, out_sigma, out_mu = tf.split(1, 3, output)
      end=0
      out_pi=tf.pack(lst[0:self._KMIX],axis=1)
      end=self._KMIX
      if self._mode==1:
        out_sigma= tf.pack(lst[end:end+self._KMIX],axis=1)
        end=end+self._KMIX
      else:
        packed=tf.pack(lst[end:end+self._KMIX*(self._n_output)],axis=1)
        out_sigma= tf.reshape(packed,shape=[-1,self._KMIX,self._n_output])#KMIX:KMIX*NDIM=24:72
        end=end+self._KMIX*(self._n_output)


      packed=tf.pack(lst[end:end+self._KMIX*(self._n_output)],axis=1)
      out_mu=tf.reshape(packed,shape=[-1,self._KMIX,self._n_output])#KMIX*NDIM:KMIX*NDIM+KMIX*NDIM=72:72+72
      # end=KMIX*(NDIM+4)

      # max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
      # out_pi = tf.sub(out_pi, max_pi)

      # out_pi = tf.exp(out_pi)

      # normalize_pi = tf.inv(tf.reduce_sum(out_pi, 1, keep_dims=True))
      # out_pi = tf.mul(normalize_pi, out_pi)
      out_pi=tf.nn.softmax(out_pi)

      out_sigma = tf.exp(out_sigma)

      return out_pi, out_sigma, out_mu

    def get_multi_normal_loss(self,final_output,y):
        """Computes the mean of negative log likelihood for P(y|x)
        y = T.matrix('y') # (minibatch_size, output_size)
        mu = T.tensor3('mu') # (minibatch_size, output_size, n_components)
        sigma = T.matrix('sigma') # (minibatch_size, n_components)
        mixing = T.matrix('mixing') # (minibatch_size, n_components)
        """
        mixing, sigma, mu = self.get_mixture_coef(final_output)
        # multivariate Gaussian
        mu=tf.transpose(mu,[0,2,1])
        exponent = -0.5 * tf.inv(sigma) * tf.reduce_sum((tf.expand_dims(y,dim=2) - mu)**2, reduction_indices=1)
        normalizer = (2.0 * np.pi * sigma)
        exponent = exponent + tf.log(mixing) - (y.get_shape()[1].value*.5)*tf.log(normalizer)
        max_exponent = tf.reduce_max(exponent,reduction_indices=1)
        mod_exponent = exponent-tf.expand_dims(max_exponent,1)
        gauss_mix = tf.reduce_sum(tf.exp(mod_exponent),reduction_indices=1)
        log_gauss = max_exponent + tf.log(gauss_mix)
        res = -tf.reduce_mean(log_gauss)
        return res

        print("okayyy")

    def get_lossfunc(self,final_output,y):
      out_pi, out_sigma, out_mu = self.get_mixture_coef(final_output)
      result_lst=[]
      # for i in range(KMIX):
      #    nrm=tf_normal(y, out_mu[:,i], out_sigma[:,i])
      #    result_lst.append(tf.mul(nrm, out_pi[:,i]))
         # if i==0:
         #    result = tf.mul(nrm, out_pi[:i])
         # else:
         #    result =result+ tf.mul(nrm, out_pi[:i])
      # result=tf.concat(1,result_lst)
      out_mu=tf.unpack(out_mu,axis=1)
      out_sigma=tf.unpack(out_sigma,axis=1)
      out_pi=tf.unpack(out_pi,axis=1)
      rslt_list=[]
      ker_rslt_list=[]
      weighted_nom_lst=[]
      for i in range(len(out_mu)):
        # result = tf_normal(y, out_mu, out_sigma)
        if self._mode==1: #bishop
          ker_result = self.tf_multi_bishop_kernel(y, out_mu[i], out_sigma[i])
          sigma=out_sigma[i]
          nominator=1
          # sigma_ndim=tf.pow(sigma,float(self._n_output))
          sigma_ndim=tf.pow(math.pi*sigma,2.0)
          nominator=tf.mul(sigma_ndim,nominator)
          weighted_nom=tf.div(out_pi[i],nominator)
          ker_rslt_list.append(ker_result)
          weighted_nom_lst.append(weighted_nom)
          # result = tf.mul(weighted_nom, result)
        # nominator=tf.div(1.0,nominator)
        # elif mode==1: #kernel
        #   result = tf_multi_bishop_normal(y, out_mu[i], out_sigma[i])
        # else:
        #   result = tf_multi_normal(y, out_mu[i], out_sigma[i])
        # result = tf_multi_normal(y, out_mu[i], out_sigma[i])
        # result = tf.mul(result, out_pi[i])

        # rslt_list.append(result)

      pack_weighted_nom=tf.pack(weighted_nom_lst)
      pack_ker_rslt=tf.pack(ker_rslt_list)
      max_ker=tf.reduce_max(pack_ker_rslt,0)
      sub_ker=tf.sub(pack_ker_rslt,max_ker)
      exp_kr=tf.exp(sub_ker)
      weighted_ker=tf.mul(pack_weighted_nom,exp_kr)

      # packed=tf.pack(rslt_list)
      result=tf.transpose(weighted_ker, [1, 0])
      # result = tf_normal(y, out_mu, out_sigma)
      # result = tf.mul(result, out_pi)
      result_sum = tf.reduce_sum(result, 1, keep_dims=True)
      result = -(tf.squeeze(tf.log(tf.maximum(result_sum,1e-10)))+max_ker)
      return tf.reduce_mean(result),result_sum,weighted_ker,pack_weighted_nom,exp_kr,pack_ker_rslt

    def generate_sample(self,final_output, selection_mode=0):
        mixing, sigma, mu = self.get_mixture_coef(final_output)

        if selection_mode==0:
            mu_lst=[]
            zp =zip(tf.unpack(mixing,axis=0),tf.unpack(mu,axis=0))
            for mx,m in zp:
                idx=tf.cast(tf.argmax(mx,0),tf.int32)
                smu=tf.gather(m,idx)
                mu_lst.append(smu)
            mu=tf.pack(mu_lst)
            return mu
        elif  selection_mode==1:
            mu_lst=[]
            zp =zip(tf.unpack(mixing,axis=0),tf.unpack(mu,axis=0))
            for mx,m in zp:
                samples = tf.multinomial(tf.log([mx]), 1) # note log-prob
                idx=tf.cast(samples[0][0], tf.int32)
                smu=tf.gather(m,idx)
                mu_lst.append(smu)
            mu=tf.pack(mu_lst)
            return mu