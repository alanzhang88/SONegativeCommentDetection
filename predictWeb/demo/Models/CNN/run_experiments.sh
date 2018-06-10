declare -a num_filters=(32 64 128 256)
declare -a filter_sizes
filter_sizes[0]="1,2,3,4"
filter_sizes[1]="1,2,3,4,5"
declare -a dp=(0.1 0.3 0.5)
declare -a lr=(0.005 0.01 0.1)
declare -a epochs=(3)
declare -a batch_size=(256 512)

for i in "${num_filters[@]}"
do
    for j in "${filter_sizes[@]}"
do 
        for k in "${dp[@]}"
do
            for l in "${lr[@]}"
do
                for m in "${epochs[@]}"
do
                    for n in "${batch_size[@]}"
do

                        python testParam.py -nf $i -fs $j -dp $k -lr $l -e $m -bs $n | tail -n 1 | tee -a ./experiments/log_$i+$j+$k+$l+$m+$n.txt 
done
done
done
done
done
done

# cd ./experiments
# python plot_graph.py