NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.

# Allow multiple threads
#export OMP_NUM_THREADS=8
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
#python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU train_prom_block.py \

python3 train_insts.py \
    --num_train_epochs 20 \
     --train_cfg_dataset "" \
    --train_dfg_dataset "" \
    --test_cfg_dataset "" \
    --test_dfg_dataset "" \
    --vocab_path "./modelout/vocab" \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --greater_is_better \
    --do_train \
    --do_eval \
    --output_dir "./modelout/" \
    --warmup_steps 10000 \
    --dataloader_num_workers 2 \
    --logging_steps 500 \
    "$@"
