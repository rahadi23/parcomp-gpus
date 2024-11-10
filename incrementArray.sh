#!/bin/bash

CTS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="./logs/incrementArray-$CTS.log"

FLOAT_SIZE=4
ONE_MB=$((1*10**6))
PER_MB_SIZE=$(($ONE_MB / $FLOAT_SIZE))

MAX_THREADS_PER_BLOCK=1024

N_MIN_IN_MB=100
N_MAX_IN_MB=1000
N_INC_IN_MB=100

N_MIN=$(($N_MIN_IN_MB * $PER_MB_SIZE))
N_MAX=$(($N_MAX_IN_MB * $PER_MB_SIZE))
N_INC=$(($N_INC_IN_MB * $PER_MB_SIZE))

BLOCK_MIN=$MAX_THREADS_PER_BLOCK
BLOCK_MAX=$MAX_THREADS_PER_BLOCK
BLOCK_INC=$MAX_THREADS_PER_BLOCK

echo $CTS >> $LOG_FILE

echo "N_MIN    : $N_MIN (${N_MIN_IN_MB}MB)" >> $LOG_FILE
echo "N_INC    : $N_INC (${N_INC_IN_MB}MB)" >> $LOG_FILE
echo "N_MAX    : $N_MAX (${N_MAX_IN_MB}MB)" >> $LOG_FILE
echo "BLOCK_MIN: $BLOCK_MIN" >> $LOG_FILE
echo "BLOCK_MAX: $BLOCK_MAX" >> $LOG_FILE
echo "BLOCK_INC: $BLOCK_INC" >> $LOG_FILE
echo "" >> $LOG_FILE

nvidia-smi >> $LOG_FILE
echo "" >> $LOG_FILE

./out/incrementArray.o $N_MIN $N_MAX $N_INC $BLOCK_MIN $BLOCK_MAX $BLOCK_INC 2>&1 >> $LOG_FILE