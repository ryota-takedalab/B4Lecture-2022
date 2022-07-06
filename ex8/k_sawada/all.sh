#! /bin/bash

for i in `seq 1 4`
do
    echo "---"
    echo "data$i"
    python main.py -i $i
done