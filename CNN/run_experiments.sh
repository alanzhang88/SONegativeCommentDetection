declare -a num_filters=(256 512)
declare -a filter_sizes
filter_sizes[0]="1,2,3,4,5"
filter_sizes[0]="3,4,5,6,7"

declare -a epochs=(4 5)

for i in "${num_filters[@]}"
do
    for j in "${filter_sizes[@]}"
do 
        for k in "${epochs[@]}"
do
            python testParam.py -nf $i -fs $j -e $k >> ./experiments/log_$i$j$k.txt
        done
    done
done

draw graph

cd ./experiments

for i in *.txt
do
    tail -n 1 $i >> results.txt
done
