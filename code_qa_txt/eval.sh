#!/bin/bash
make

nhop_subg=3
dataset=vanilla
data_root=../nips_data
#net_type=NetMultiHop
net_type=NetLatentY

result_root=$HOME/scratch/results/graph_mem/nhop-$nhop_subg/$dataset

num_neg=10000
max_bp_iter=1
max_q_iter=3
batch_size=128
n_hidden=64
n_embed=256
margin=0.1
learning_rate=0.01
max_iter=4000000
cur_iter=800
w_scale=0.01
loss_type=cross_entropy
save_dir=$result_root/embed-$n_embed

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

./build/main \
    -test_only 1 \
    -test_topk 5 \
    -num_neg $num_neg \
    -loss_type $loss_type \
    -data_root $data_root \
    -dataset $dataset \
    -n_hidden $n_hidden \
    -nhop_subg $nhop_subg \
    -lr $learning_rate \
    -max_bp_iter $max_bp_iter \
    -net_type $net_type \
    -max_q_iter $max_q_iter \
    -margin $margin \
    -max_iter $max_iter \
    -svdir $save_dir \
    -embed $n_embed \
    -batch_size $batch_size \
    -m 0.9 \
    -l2 0.00 \
    -w_scale $w_scale \
    -int_report 1 \
    -int_test 1 \
    -int_save 1000000 \
    -cur_iter $cur_iter
