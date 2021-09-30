#!/bin/sh

python ./eval/convert.py $1 $2
python ./eval/task3_scorer_onefile.py -s ./eval/official_prediction.txt -r ./dataset/protechn_corpus_eval/test -t ./dataset/protechn_corpus_eval/propaganda-techniques-names.txt

