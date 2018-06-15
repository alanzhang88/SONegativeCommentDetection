import argparse
from CNNutil import CNNModel
import json
import sys


###Experiments on hyperparameter combination

parser = argparse.ArgumentParser()
parser.add_argument('-nf',dest='num_filters',type=int,required=False)
parser.add_argument('-fs',dest='filter_sizes',type=str,required=False)
parser.add_argument('-dp', dest="drop_prob", type=float, required=False)
parser.add_argument('-lr', dest="lr", type=float, required=False)
parser.add_argument('-bs', dest="batch_size", type=int, required=False)
parser.add_argument('-a', dest="activation", type=str, required=False)
parser.add_argument('-c', dest="count", type=int, required=True)


args = parser.parse_args()
count = args.count

with open("parameter_temp3.json", 'r') as jsondata:
    data = json.loads(jsondata.read())


data[count] = {}

if args.num_filters is not None:
    num_filters = args.num_filters
    data[count]["num_filters"] = num_filters
if args.filter_sizes is not None:
    if len(args.filter_sizes) > 1:
        filter_sizes = [int(x) for x in args.filter_sizes.split(",")]
    else:
        filter_sizes = [int(args.filter_sizes)]
    data[count]["filter_sizes"] = filter_sizes
if args.drop_prob is not None:
    dp = args.drop_prob
    data[count]["drop_prob"] = dp
if args.lr is not None:
    lr = args.lr
    data[count]["lr"] = lr
if args.batch_size is not None:
    bs = args.batch_size
    data[count]["batch_size"] = bs
if args.activation is not None:
    activation = args.activation
    data[count]["activation"] = activation


model = CNNModel(save_model=True)

for key, value in data[count].items():
    model.model = None
    setattr(model, key, value)

history = model.build_model()
data[count]["recall"] = history.history['val_TNR'][2]
data[count]['precision'] = history.history['val_precision'][2]
data[count]['f1'] = history.history['val_f1_score'][2]
data[count]['accuracy'] = history.history['val_acc'][2]
data[count]['tn'] = history.history['val_TN'][2]
data[count]['fp'] = history.history['val_FP'][2]
data[count]['fn'] = history.history['val_FN'][2]
data[count]['tp'] = history.history['val_TP'][2]

f = open("parameter_temp3.json", "w")
f.write(json.dumps(data))