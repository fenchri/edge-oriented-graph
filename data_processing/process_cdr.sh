#!/usr/bin/env bash

for d in "Training" "Development" "Test";
do
    python3 process.py --input_file ../data/CDR/CDR.Corpus.v010516/CDR_${d}Set.PubTator.txt \
                       --output_file ../data/CDR/processed/${d} \
                       --data CDR

    python3 filter_hypernyms.py --mesh_file 2017MeshTree.txt \
                                --input_file ../data/CDR/processed/${d}.data \
                                --output_file ../data/CDR/processed/${d}_filter.data
done

mv ../data/CDR/processed/Training.data ../data/CDR/processed/train.data
mv ../data/CDR/processed/Development.data ../data/CDR/processed/dev.data
mv ../data/CDR/processed/Test.data ../data/CDR/processed/test.data

mv ../data/CDR/processed/Training_filter.data ../data/CDR/processed/train_filter.data
mv ../data/CDR/processed/Development_filter.data ../data/CDR/processed/dev_filter.data
mv ../data/CDR/processed/Test_filter.data ../data/CDR/processed/test_filter.data

# merge train and dev
cat ../data/CDR/processed/train_filter.data > ../data/CDR/processed/train+dev_filter.data
cat ../data/CDR/processed/dev_filter.data >> ../data/CDR/processed/train+dev_filter.data



