ROOT_DIR=/share/zhaohanyu/Code/llm_rlhf

# DeepSpeed Team

OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./checkpoints/output_step1_llama2-7b_sftv925
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --hostfile ${ROOT_DIR}/config/step1hostfile main2.py \
    --deepspeed \
    --project_name rlhf_llama3 \
    --experiment_name llama3_sft \
    --data_path /share/zhaohanyu/Data/SFT/sft_v925/input \
    --model_name_or_path /share/models/llama2_7b_hf \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_interval 1000 \
    --eval_interval 500 \
    --max_seq_len 1024 \
    --learning_rate 9.65e-6 \
    --loss_scale 0.0 \
    --loss_scale_window 100 \
    --weight_decay -1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps -1 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --output_dir $OUTPUT \
    --part_data_size -1 \
    | tee $OUTPUT/training.log \

