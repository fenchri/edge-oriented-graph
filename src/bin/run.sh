#!/bin/bash

expA=("ME MS ES SS-ind"
      "MM MS ES SS-ind"
      "MM ME ES SS-ind"
      "MM ME MS SS-ind"
      "MM ME MS ES"
      "MM ME MS ES SS-ind"
      "MM ME"
      "ES SS-ind")

#options=("--context"
#         "--types"
#         "--distances"
#         "--context --types"
#         "--context --distances"
#         "--distances --types")


options=("--types --dist")


for o in "${options[@]}":
do
    counter=0
    for w in 1 2 3 4;
    do
        # EOG + SS-ind
        python3 eog.py --config ../configs/parameters_cdr.yaml --train \
                                                               --edges MM ME MS ES SS-ind \
                                                               --walks ${w} "${o}" \
                                                               --gpu ${counter} &
        counter=$((counter+1))

        # EOG + SS
        python3 eog.py --config ../configs/parameters_cdr.yaml --train \
                                                               --edges MM ME MS ES SS \
                                                               --walks ${w} "${o}" \
                                                               --gpu ${counter} &
        counter=$((counter+1))
    done
#    wait
done


wait


for w in 1 2 3 4;
do
    # rest
    counter=0
    for edg in "${expA[@]}";
    do
        python3 eog.py --config ../configs/parameters_cdr.yaml --train \
                                                               --edges "${edg}" \
                                                               --walks ${w} "${o}" \
                                                               --gpu ${counter} &
        counter=$((counter+1))
    done
    wait
done


# fully connected
for w in 1 2 3 4;
do
    counter=0
    for edg in "${expA[@]}";
    do
        python3 eog.py --config ../configs/parameters_cdr.yaml --train \
                                                               --edges "FULL" \
                                                               --walks ${w} "${o}" \
                                                               --gpu ${counter} &
        counter=$((counter+1))
    done
done


# sentence
for w in 1 2 3 4;
do
    counter=0
    for edg in "${expA[@]}";
    do
        python3 eog.py --config ../configs/parametes_cdr.yaml --train \
                                                              --edges MM ME MS ES SS-ind \
                                                              --walks ${w} \
                                                              --gpu ${counter} \
                                                              --walks ${w} "${o}" \
                                                              --window 1 &
        counter=$((counter+1))
    done
done

wait



# no inference
#counter=0
#for w in 0 1 2 3 4 5;
#do
#    python3 eog.py --config ../configs/parameters_cdr.yaml --train \
#                                                           --edges "EE" \
#                                                           --walks ${w} \
#                                                           --gpu ${counter} &
#    counter=$((counter+1))
#done


# sentence (different options, EOG model)
#python3 eog.py --config ../configs/parameters_cdr.yaml --train \
#                                                       --edges MM ME MS ES SS-ind \
#                                                       --walks 3 \
#                                                       --types \
#                                                       --context \
#                                                       --window 1 \
#                                                       --gpu 0 &
#
#python3 eog.py --config ../configs/parameters_cdr.yaml --train \
#                                                       --edges MM ME MS ES SS-ind \
#                                                       --walks 3 \
#                                                       --types \
#                                                       --window 1 \
#                                                       --gpu 1 &
#
#python3 eog.py --config ../configs/parameters_cdr.yaml --train \
#                                                       --edges MM ME MS ES SS-ind \
#                                                       --walks 3 \
#                                                       --context \
#                                                       --window 1 \
#                                                       --gpu 2 &
#
#python3 eog.py --config ../configs/parameters_cdr.yaml --train \
#                                                       --edges MM ME MS ES SS-ind \
#                                                       --walks 3 \
#                                                       --window 1 \
#                                                       --gpu 3 &


