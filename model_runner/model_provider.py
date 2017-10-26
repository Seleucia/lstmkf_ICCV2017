from model_runner.cnn import inception_eval
from model_runner.cnn import inception_train
# import  inception_eval
# import inception_train
# import vgg_train

from model_runner.cnn import vgg_eval


def get_model(params):
    if(params["model"]=="inception_v1"):
        model_eval = inception_eval
        model_train = inception_train
        params['scope']='InceptionV1'
        params['checkpoint_exclude_scopes']=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
    elif(params["model"]=="inception_v2"):
        params['scope']='InceptionV2'
        params['model_file']=params["model_file"]+'/'+"inception_v2.ckpt"
        params['checkpoint_exclude_scopes']=["InceptionV2/Logits", "InceptionV2/AuxLogits"]
    elif(params["model"]=="inception_v3"):
        model_eval = inception_eval
        model_train = inception_train
        params['scope']='InceptionV3'
        params['model_file']=params["model_file"]+'/'+"inception_v3.ckpt"
        params['checkpoint_exclude_scopes']=["InceptionV3/Logits", "InceptionV3/AuxLogits"]
    elif(params["model"]=="inception_v3"):
        model_eval = inception_eval
        model_train = inception_train
        params['scope']='InceptionV3'
        params['model_file']=params["model_file"]+'/'+"inception_v3.ckpt"
        params['checkpoint_exclude_scopes']=["InceptionV3/Logits", "InceptionV3/AuxLogits"]
    elif(params["model"]=="inception_resnet_v2"):
        model_eval = inception_eval
        model_train = inception_train
        params['scope']='InceptionResnetV2'
        # params['model_file']=params["model_file"]+"inception_resnet_v2_2016_08_30.ckpt"
        params['checkpoint_exclude_scopes']=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    elif(params["model"]=="resnet_v1"):
        model_eval = inception_eval
        model_train = inception_train
    elif(params["model"]=="resnet_v1"):
        model_eval = inception_eval
        model_train = inception_train
    elif(params["model"]=="alexnet"):
        model_eval = inception_eval
        model_train = inception_train
    elif(params["model"]=="vgg19"):
        model_eval = vgg_eval
        model_train = vgg_train
        params['model_file']=params["model_file"]+'/'+"vgg_19.ckpt"
        params['scope']='vgg_19'
        params['checkpoint_exclude_scopes']=["vgg_19/fc8"]
    elif(params["model"]=="vgg16"):
        model_eval = vgg_eval
        model_train = vgg_train
        params['model_file']=params["model_file"]+'/'+"vgg_16.ckpt"
        params['scope']='vgg_16'
        params['checkpoint_exclude_scopes']=["vgg_16/fc8"]
    else:
        raise Exception('Wrong model calling....') #
    return (model_train,model_eval,params)
