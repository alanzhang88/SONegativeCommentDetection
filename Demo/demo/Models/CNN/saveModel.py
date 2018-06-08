from keras.callbacks import Callback

class SaveModel(Callback):
    def __init__(self,target_name,target_val,validation_data=()):
        super(Callback,self).__init__()
        self.validation_data = validation_data
        self.target_val = target_val
        self.target_name = target_name
        self.last = -1

    def on_epoch_end(self,epoch,logs={}):
        res = self.model.test_on_batch(self.validation_data[0],self.validation_data[1])
        for i in range(len(self.model.metrics_names)):
            if self.model.metrics_names[i] == self.target_name and res[i] >= self.target_val and res[i] > self.last:
                self.last = res[i]
                print('Saveing model with %s reaching %f' % (self.target_name,res[i]))
                self.model.save(filepath='./CNNmodel.h5')
