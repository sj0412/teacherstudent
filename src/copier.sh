#!/bin/bash

id='07184.png'
sources='/mnt/dev/datasets/gated2depth/'
target='/mnt/dev/projects/gjm/DPT_Gated_trianer_new/figs/input_night_test_samples/'
cp "${sources}"data/real/gated0_10bit/"${id}" "${target}"gated0_10bit/"${id}"

cp "${sources}"data/real/gated1_10bit/"${id}" "${target}"gated1_10bit/"${id}"

cp "${sources}"data/real/gated2_10bit/"${id}" "${target}"gated2_10bit/"${id}"

#cp -r  /home/xt/PycharmProjects/trianingModule/src/input_day/* /home/xt/PycharmProjects/trianingModule/input/
echo "complete!"
