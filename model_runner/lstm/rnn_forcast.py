from model_runner.lstm_forcast.malik_erd_lstm import  Model

def forcast(params):
    seed_length=params['seed_length']
    forcast_length=params['forcast_length']
    params['inp_sequence_length']=seed_length
    params['out_sequence_length']=seed_length+forcast_length
    model_forcast = Model(params)