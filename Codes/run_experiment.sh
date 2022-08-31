#!/bin/bash

for ((i=0; i<=19; i+=1)); do
	python mnist.py --extra-sample false --epochs 1 --seed $i
        python mnist.py --extra-sample true --epochs 1 --seed $i
done