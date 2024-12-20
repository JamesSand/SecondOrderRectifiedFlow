#!/bin/bash

set -e

FIRST_ORDER_WEIGHT=1.0

VERSIONS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

SECOND_ORDER_WEIGHTS=(1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13)

for VERSION in "${VERSIONS[@]}"
do
    for SECOND_ORDER_WEIGHT in "${SECOND_ORDER_WEIGHTS[@]}"
    do
        SAVE_DIR_NAME="${FIRST_ORDER_WEIGHT}_${SECOND_ORDER_WEIGHT}_V${VERSION}"

        python train_gpu.py \
            --first_order_weight $FIRST_ORDER_WEIGHT \
            --second_order_weight $SECOND_ORDER_WEIGHT \
            --save_dir_name $SAVE_DIR_NAME 
    done
done
