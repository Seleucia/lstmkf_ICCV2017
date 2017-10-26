import utils as ut
import  numpy as np


def prepare_batch(is_test,index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst):
    if params["model"]=="lstm":
        return prepare_lstm_batch(is_test,index_list, minibatch_index, batch_size, S_list,dic_state, params, Y, X,R_L_list,F_list,state_reset_counter_lst)

    else:
        return prepare_kfl_QRFf_batch(is_test,index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst)



def prepare_lstm_batch(is_test,index_list, minibatch_index, batch_size, S_list,dic_state, params, Y, X,R_L_list,F_list,state_reset_counter_lst):
    #index_list= list of ids for sequences..
    #LStateList current states
    #LStateList_pre previus batch states
    #repeat list with in a single sequnce.
    if is_test==1:
        reset_state = -1
    else:
        reset_state = params['reset_state']

    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024

    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_S=ut.get_zero_state(params)

    if(minibatch_index>0):
        for idx in range(batch_size):
            state_reset_counter=state_reset_counter_lst[idx]
            if state_reset_counter%reset_state==0 and reset_state>0:
                # print 'resetting state: reset counter'
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif(pre_sid[idx]!=curr_sid[idx]):# if sequence changed reset state also
                # print 'resetting state: sequence change'
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx]=0
            elif (curr_id_lst[idx]==pre_id_lst[idx]): #if we repeated the value we should repeat state also.
                # print 'repeate state'
                state_reset_counter_lst[idx]=state_reset_counter-1
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=dic_state["lstm_pre"][s][0][idx]
                    new_S[s][1][idx]=dic_state["lstm_pre"][s][1][idx]
                    # new_S[s][idx,:]=LStateList_pre[s][idx,:]
            else: #LSTM will take the last state....
                for s in range(params['nlayer']):
                    new_S[s][0][idx]=dic_state["lstm_t"][s][0][idx]
                    new_S[s][1][idx]=dic_state["lstm_t"][s][1][idx]
                    # new_S[s][idx,:]=LStateList[s][idx,:]
    x=X[curr_id_lst]
    y=Y[curr_id_lst]
    if R_L_list != None:
        r=R_L_list[curr_id_lst]
    else:
        r=None
    f=F_list[curr_id_lst]
    dic_state["lstm_pre"]=new_S


    return (dic_state,x,y,r,f,curr_sid,state_reset_counter_lst,curr_id_lst)


def prepare_kfl_QRFf_batch(is_test,index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst):
    #index_list= list of ids for sequences..
    #LStateList current states
    #LStateList_pre previus batch states
    #repeat list with in a single sequnce.
    if is_test==1:
        reset_state = -1
        P_mul=0.01
    else:
        reset_state = params['reset_state']
        P_mul=params['P_mul']



    curr_id_lst=index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
    pre_id_lst=index_list[(minibatch_index-1) * batch_size:(minibatch_index) * batch_size]
    curr_sid=S_list[curr_id_lst]
    pre_sid=S_list[pre_id_lst]
    new_state_F=ut.get_zero_state(params)
    x=X[curr_id_lst]
    y=Y[curr_id_lst]
    new_P = np.asarray([np.diag([1.0] * params['n_output']) for i in range(params["batch_size"])],dtype=np.float32) * P_mul
    _x=np.copy(x[:,0,:])
    if "Q" in params["model"]:
        new_state_Q=ut.get_zero_state(params,'Q')
    if "R" in params["model"]:
        new_state_R=ut.get_zero_state(params,'R')
    if "K" in params["model"]:
        new_state_K=ut.get_zero_state(params,'K')

    if(minibatch_index>0):
        for idx in range(batch_size):
            state_reset_counter=state_reset_counter_lst[idx]
            if state_reset_counter%reset_state==0 and reset_state>0:
                for s in range(params['nlayer']):
                    new_state_F[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_state_F[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                if "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx]=np.zeros(shape=(1,params['Qn_hidden']), dtype=np.float32)
                        new_state_Q[s][1][idx]=np.zeros(shape=(1,params['Qn_hidden']), dtype=np.float32)
                if "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx]=np.zeros(shape=(1,params['Rn_hidden']), dtype=np.float32)
                        new_state_R[s][1][idx]=np.zeros(shape=(1,params['Rn_hidden']), dtype=np.float32)
                if "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx]=np.zeros(shape=(1,params['Kn_hidden']), dtype=np.float32)
                        new_state_K[s][1][idx]=np.zeros(shape=(1,params['Kn_hidden']), dtype=np.float32)
                new_P = np.asarray([np.diag([1.0] * params['n_output']) for i in range(params["batch_size"])])*P_mul
                _x[idx] = dic_state["_x_pre"][idx]
                state_reset_counter_lst[idx]=0
            elif(pre_sid[idx]!=curr_sid[idx]):# if sequence changed reset state also
                for s in range(params['nlayer']):
                    new_state_F[s][0][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                    new_state_F[s][1][idx]=np.zeros(shape=(1,params['n_hidden']), dtype=np.float32)
                if "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx]=np.zeros(shape=(1,params['Qn_hidden']), dtype=np.float32)
                        new_state_Q[s][1][idx]=np.zeros(shape=(1,params['Qn_hidden']), dtype=np.float32)
                if "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx]=np.zeros(shape=(1,params['Rn_hidden']), dtype=np.float32)
                        new_state_R[s][1][idx]=np.zeros(shape=(1,params['Rn_hidden']), dtype=np.float32)
                if "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx]=np.zeros(shape=(1,params['Kn_hidden']), dtype=np.float32)
                        new_state_K[s][1][idx]=np.zeros(shape=(1,params['Kn_hidden']), dtype=np.float32)
                new_P[idx]=np.diag([1.0] * params['n_output'])* P_mul

                state_reset_counter_lst[idx]=0
            elif (curr_id_lst[idx]==pre_id_lst[idx]): #if the item in sequence repeated, we should repeat state also.
                state_reset_counter_lst[idx]=state_reset_counter-1
                for s in range(params['nlayer']):
                    new_state_F[s][0][idx]=dic_state["F_pre"][s][0][idx]
                    new_state_F[s][1][idx]=dic_state["F_pre"][s][1][idx]
                if "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx]=dic_state["Q_pre"][s][0][idx]
                        new_state_Q[s][1][idx]=dic_state["Q_pre"][s][1][idx]
                if "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx]=dic_state["R_pre"][s][0][idx]
                        new_state_R[s][1][idx]=dic_state["R_pre"][s][1][idx]
                if "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx]=dic_state["K_pre"][s][0][idx]
                        new_state_K[s][1][idx]=dic_state["K_pre"][s][1][idx]
                    # new_S[s][idx,:]=LStateList_pre[s][idx,:]
                new_P[idx] = dic_state["PCov_pre"][idx]
                _x[idx] = dic_state["_x_pre"][idx]
            else: #This condition initialise the state with the last state of the model
                for s in range(params['nlayer']):
                    new_state_F[s][0][idx]=dic_state["F_t"][s][0][idx]
                    new_state_F[s][1][idx]=dic_state["F_t"][s][1][idx]
                if "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx]=dic_state["Q_t"][s][0][idx]
                        new_state_Q[s][1][idx]=dic_state["Q_t"][s][1][idx]
                if "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx]=dic_state["R_t"][s][0][idx]
                        new_state_R[s][1][idx]=dic_state["R_t"][s][1][idx]
                if "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx]=dic_state["K_t"][s][0][idx]
                        new_state_K[s][1][idx]=dic_state["K_t"][s][1][idx]
                    # new_S[s][idx,:]=LStateList[s][idx,:]
                new_P[idx] = dic_state["PCov_t"][idx]
                _x[idx]=dic_state["_x_t"][idx] #we use the last prediction values....

    dic_state["F_pre"]=new_state_F
    dic_state["PCov_pre"]=new_P
    if "Q" in params["model"]:
        dic_state["Q_pre"]=new_state_Q
    if "R" in params["model"]:
        dic_state["R_pre"]=new_state_R
    if "K" in params["model"]:
        dic_state["K_pre"]=new_state_K

    dic_state["_x_pre"]=_x

    if R_L_list != None:
        r=R_L_list[curr_id_lst]
    else:
        r=None
    f=F_list[curr_id_lst]

    return (dic_state,x,y,r,f,curr_sid,state_reset_counter_lst,curr_id_lst)
def get_feed(model,params,r,x,y,I,dic_state, is_training=0):
    if is_training==1:
        if params["model"] == "kfl_K":
            feed = {model._z: x, model.target_data: y, model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                    , model.initial_state_K: dic_state["K_pre"], model.output_keep_prob: params['rnn_keep_prob']}
        elif params["model"] == "kfl_QRf":
            feed = {model._z: x, model.target_data: y, model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                    , model.initial_state_Q_noise: dic_state["Q_pre"], model.initial_state_R_noise: dic_state["R_pre"]
                    , model._P_inp: dic_state["PCov_pre"],model._x_inp: dic_state["_x_pre"],
                    model._I: I, model.output_keep_prob: params['rnn_keep_prob'],
                    model.input_keep_prob: params['input_keep_prob']}
        elif params["model"] == "kfl_Rf":
            feed = {model._z: x, model.target_data: y, model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                    , model.initial_state_R_noise: dic_state["R_pre"], model._P_inp: dic_state["PCov_pre"],
                    model._I: I, model.output_keep_prob: params['rnn_keep_prob'],model.input_keep_prob: params['input_keep_prob']}
        elif params["model"] == "kfl_f":
            feed = {model._z: x, model.target_data: y, model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                    , model._P_inp: dic_state["PCov_pre"],model._I: I, model.output_keep_prob: params['rnn_keep_prob'],
                    model.input_keep_prob: params['input_keep_prob']}
        elif params["model"] == "kfl_QRFf":
            feed = {model._z: x, model.target_data: y, model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                    , model.initial_state_Q_noise: dic_state["Q_pre"], model.initial_state_R_noise: dic_state["R_pre"]
                , model._P_inp: dic_state["PCov_pre"], model._I: I, model.output_keep_prob: params['rnn_keep_prob']}
        elif params["model"] == "lstm":
            feed = {model.input_data: x, model.input_zero:np.ceil(x), model.target_data: y,
                        model.initial_state: dic_state["lstm_pre"],model.repeat_data: r,model.is_training:True,
                        model.output_keep_prob:params['rnn_keep_prob']}
        elif params["model"] == "kf_QR":
            feed = {model.input_data: x, model.target_data: y, model.repeat_data: r,model.initial_state_Q_noise: dic_state["Q_pre"],
                            model.initial_state_R_noise: dic_state["R_pre"], model.H: params['H_mat'], model.F: params['F_mat'],model.output_keep_prob:params['rnn_keep_prob']}
    else:
        if params["model"]=="kfl_K":
            feed = {model._z: x, model.target_data: y,model.repeat_data: r, model.initial_state: dic_state["F_pre"]
            , model.initial_state_K: dic_state["K_pre"],model.output_keep_prob:1.0}
        elif params["model"]=="kfl_QRf":
            feed = {model._z: x, model.target_data: y,model.repeat_data: r, model.initial_state: dic_state["F_pre"]
            , model.initial_state_Q_noise: dic_state["Q_pre"], model.initial_state_R_noise:dic_state["R_pre"]
                , model._P_inp: dic_state["PCov_pre"],model._x_inp: dic_state["_x_pre"], model._I: I,model.output_keep_prob:1.0,
                    model.input_keep_prob: 1}
        elif params["model"]=="kfl_Rf":
            feed = {model._z: x, model.target_data: y,model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                , model.initial_state_R_noise:dic_state["R_pre"], model._P_inp: dic_state["PCov_pre"],
                    model._I: I,model.output_keep_prob:1.0,model.input_keep_prob: 1}
        elif params["model"]=="kfl_f":
            feed = {model._z: x, model.target_data: y,model.repeat_data: r, model.initial_state: dic_state["F_pre"]
                , model._P_inp: dic_state["PCov_pre"], model._I: I,model.output_keep_prob:1.0,model.input_keep_prob: 1}
        elif params["model"]=="kfl_QRFf":
            feed = {model._z: x, model.target_data: y,model.repeat_data: r, model.initial_state: dic_state["F_pre"]
            , model.initial_state_Q_noise: dic_state["Q_pre"], model.initial_state_R_noise:dic_state["R_pre"]
                , model._P_inp: dic_state["PCov_pre"], model._I: I,model.output_keep_prob:1.0}
        elif params["model"] == "lstm":
            feed = {model.input_data: x, model.target_data: y, model.initial_state: dic_state["lstm_pre"],
                    model.repeat_data: r ,model.is_training:False,model.output_keep_prob:1.0}
        elif params["model"] == "kf_QR":
            feed = {model.input_data: x, model.target_data: y, model.repeat_data: r,model.initial_state_Q_noise: dic_state["Q_pre"],
                    model.initial_state_R_noise: dic_state["R_pre"], model.H: params['H_mat'], model.F: params['F_mat'],model.output_keep_prob:1.0}
    return feed

