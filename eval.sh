#!/bin/bash
ROLLNO=$1
jupyter nbconvert --to python "${ROLLNO}_foml24_hackathon.ipynb" --output "${ROLLNO}_foml24_hackathon"
python "${ROLLNO}_foml24_hackathon.py" --train-file train.csv --test-file test.csv --predictions-file ${ROLLNO}_submission.csv
