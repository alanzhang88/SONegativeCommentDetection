import argparse
from CNNutil import CNNModel

#Test the impact of single parameter change

parser = argparse.ArgumentParser()
parser.add_argument('-f',dest='fieldName',help='the name of the hyperparameter used to test, need to exactly match the field name of the object',type=str,required=True)
parser.add_argument('-v',dest='fieldValue',help='the value of the hyperparameter, input as list eg [32,64,128] for num_filters and [[2,3,4],[3,4,5]] for filter_sizes',type=str,required=True)
parser.add_argument('-r',dest='random_state',help='set the random state of data data_generator', type=int)
parser.add_argument('--graph-var',dest='graph',type=str,help='the hyperparameter name that will be used as the variables to graph against accuracy',required=True)
args = parser.parse_args()


fieldName = args.fieldName
fieldValue = eval(args.fieldValue)
random_state = args.random_state if args.random_state else None

model = CNNModel(save_model=False)

for i in fieldValue:
    print('Begin Test Condition %s=%s' % (fieldName,str(i)))
    model.model = None
    setattr(model,fieldName,i)
    model.build_model()
    print('End Test Condition %s=%s' % (fieldName,str(i)))

