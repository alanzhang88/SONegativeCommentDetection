from CNNutil import CNNModel

filter_size = list(range(2,11))
model = CNNModel(save_model=False)
for i in filter_size:
    print('Begin Test Condition filter_size=%d' % i)
    model.model = None
    model.filter_sizes = [i]
    model.build_model()
    print('End Test Condition filter_size=%d' % i)
