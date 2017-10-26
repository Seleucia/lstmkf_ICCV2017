path="/home/coskun/PycharmProjects/poseftv4/files/logs/lstmv2_2layer_pretrained_2_12-14-26-848264.txt"

def load_data():
    with open(path,mode='r') as f:
        lines=f.readlines()
        x=[float(line.split(' ')[-1].replace('\n','')) for line in lines if  'TRAINING_Data Loss' in line]
        y=[float(line.split(' ')[-1].replace('\n','')) for line in lines if  'TEST_Data Loss' in line]
        titel=[line.split(':')[-1].replace('\n','') for line in lines if  'Deployment notes' in line][0]
    return x,y,titel

def plot_error():
    ax = plt.subplot(111)

    training,test,title= load_data()
    epoch_lst=range(len(test))
    plt.plot(epoch_lst,training,label='Training')
    plt.plot(test,label='Test')
    plt.xticks(np.arange(1, max(epoch_lst)+1, 1.0))
    plt.yticks(np.arange(0, 0.1, 0.005))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(title)
    ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )



    plt.show()

plot_error()