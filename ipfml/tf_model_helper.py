# module filewhich contains helpful display function

# avoid tk issue
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

def save(history, filename):
    '''
    @brief Function which saves data from neural network model
    @param history : tensorflow model history
    @param filename : information about model filename
    @return nothing
    '''
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str('%s_accuracy.png' % filename))

    # clear plt history
    plt.gcf().clear()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str('%s_loss.png' % filename))

def show(history, filename):
    '''
    @brief Function which shows data from neural network model
    @param history : tensorflow model history
    @param filename : information about model filename
    @return nothing
    '''
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
