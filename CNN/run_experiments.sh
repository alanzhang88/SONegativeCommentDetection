declare -a num_filters=(32 64 128 256)
declare -a filter_sizes
# filter_sizes[0]="1,2,3,4"
filter_sizes[1]="4"
declare -a dp=(0.1 0.3 0.5)
declare -a lr=(0.001 0.002 0.003)
declare -a batch_size=(32)
declare -a activation=('tanh' 'softmax' 'relu' 'sigmoid')

declare -a count
count=0
for i in "${num_filters[@]}"
do
    for j in "${filter_sizes[@]}"
do 
        for k in "${dp[@]}"
do
            for l in "${lr[@]}"
do
                for m in "${activation[@]}"
do
                    for n in "${batch_size[@]}"
do
                        count=$((count + 1))
                        # python testParam.py -nf $i -fs $j -dp $k -lr $l -a $m -bs $n count | tail -n 1 | tee -a ./experiments2/log_$i+$j+$k+$l+$m+$n.txt
                        python testParam.py -nf $i -fs $j -dp $k -lr $l -a $m -bs $n -c $count 
                    done
                done
            done
        done
    done
done

# cd ./experiments
# python plot_graph.py