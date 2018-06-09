declare -a num_filters=(32 64)
declare -a filter_sizes
filter_sizes[0]="1,2,3,4"


declare -a epochs=(3)

for i in "${num_filters[@]}"
do
    for j in "${filter_sizes[@]}"
do 
        for k in "${epochs[@]}"
do
            python testParam.py -nf $i -fs $j -e $k | tail -n 1 | tee -a ./experiments/log_$i+$j+$k.txt 
        done
    done
done


cd ./experiments
python plot_graph.py