import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-f',dest='fieldName',help='the name of the hyperparameter used to test, need to exactly match the field name of the object',type=str,required=True)
# parser.add_argument('-v',dest='fieldValue',help='the value of the hyperparameter, input as list eg [32,64,128] for num_filters and [[2,3,4],[3,4,5]] for filter_sizes',type=str,required=True)
# parser.add_argument('-r',dest='random_state',help='set the random state of data data_generator', type=int)
# args = parser.parse_args()
parser.add_argument('-nf',dest='num_filters',type=int,required=False)
parser.add_argument('-fs',dest='filter_sizes',type=str,required=False)
parser.add_argument('-e',dest='epochs',type=int,required=False)
parser.add_argument('-dp', dest="drop_prob", type=float, required=False)
parser.add_argument('-lr', dest="lr", type=float, required=False)
parser.add_argument('--graph-var',dest='graph',type=str,help='the hyperparameter name that will be used as the variables to graph against accuracy',required=True)

args = parser.parse_args()
# num_filters = args.num_filters if args.num_filters is not None
# filter_sizes =[int(x) for x in args.filter_sizes.split(",")] if filter_sizes is not None
# epochs = args.epochs if epochs is not None
# drop_prob = args.drop_prob if args.drop_prob is not None
# lr = args.lr if args.lr is not None
from CNNutil import CNNModel
params = {}
if args.num_filters is not None:
    num_filters = args.num_filters
    params["num_filters"] = num_filters
if args.filter_sizes is not None:
    filter_sizes = [int(x) for x in args.filter_sizes.split(",")]
    params["filter_sizes"] = filter_sizes
if args.epochs is not None:
    epochs = args.epochs
    params["epochs"] = epochs

# params = {"num_filters": num_filters, "filter_sizes": filter_sizes, "epochs": epochs}

# fieldName = args.fieldName
# fieldValue = eval(args.fieldValue)
# random_state = args.random_state if args.random_state else None

model = CNNModel(save_model=False)
# for i in fieldValue:
#     print('Begin Test Condition %s=%s' % (fieldName,str(i)))
#     model.model = None
#     setattr(model,fieldName,i)
#     model.build_model()
#     print('End Test Condition %s=%s' % (fieldName,str(i)))

for key, value in params.items():
    model.model = None
    setattr(model, key, value)

print('Begin Test Condition %s=%s' % (args.graph,str(params.get(args.graph))))
model.build_model()
print('End Test Condition %s=%s' % (args.graph,str(params.get(args.graph))))
