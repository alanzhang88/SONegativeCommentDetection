declare -a num_filters=(32  64)
declare -a filter_sizes

filter_sizes[0]="1,2,3"
filter_sizes[1]="1,2,3,4"
# filter_sizes[2]="1 2 3 4 5"
# filter_sizes[3]="1 2 3 4 5 6"
# filter_sizes[4]="1 2 3 4 5 6 7"

declare -a epochs=(3 4)

count=0
for i in "${num_filters[@]}"
do
    for j in "${filter_sizes[@]}"
do 
        for k in "${epochs[@]}"
do
            ((count++))
            python testParam.py -nf $i -fs $j -e $k >> log.txt
        done
    done
done