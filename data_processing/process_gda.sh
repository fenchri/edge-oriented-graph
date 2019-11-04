#!/usr/bin/env bash

for d in "training" "testing";
do
    python3 gda2pubtator.py --input_folder ../data/GDA/${d}_data/ \
                            --output_file ../data/GDA/processed/${d}.pubtator

    python3 process.py --input_file ../data/GDA/processed/${d}.pubtator \
                       --output_file ../data/GDA/processed/${d} \
                       --data GDA
done

mv ../data/GDA/processed/testing.data ../data/GDA/processed/test.data
mv ../data/GDA/processed/training.data ../data/GDA/processed/train+dev.data

python3 split_gda.py --input_file ../data/GDA/processed/train+dev.data \
                     --output_train ../data/GDA/processed/train.data \
                     --output_dev ../data/GDA/processed/dev.data \
                     --list train_gda_docs
