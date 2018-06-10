import argparse

###Experiments on hyperparameter combination

parser = argparse.ArgumentParser()
parser.add_argument('-nf',dest='num_filters',type=int,required=False)
parser.add_argument('-fs',dest='filter_sizes',type=str,required=False)
parser.add_argument('-e',dest='epochs',type=int,required=False)
parser.add_argument('-dp', dest="drop_prob", type=float, required=False)
parser.add_argument('-lr', dest="lr", type=float, required=False)
parser.add_argument('-bs', dest="batch_size", type=int, required=False)


args = parser.parse_args()
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
if args.drop_prob is not None:
    dp = args.drop_prob
    params["drop_prob"] = dp
if args.lr is not None:
    lr = args.lr
    params["lr"] = lr
if args.batch_size is not None:
    bs = args.batch_size
    params["batch_size"] = bs


model = CNNModel(save_model=False)

for key, value in params.items():
    model.model = None
    setattr(model, key, value)

model.build_model()
