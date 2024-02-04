import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import conf as co
import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments
)
from swift.llm.run import infer_main, sft_main

model_type = ModelType.llama2_7b_chat

sft_args = SftArguments(
    model_type=model_type,
    model_cache_dir=co.base_model_path,
    eval_steps=500,
    train_dataset_sample=-1,
    num_train_epochs=2,
    batch_size=1,
    max_length=4096,
    max_new_tokens=4096,
    use_flash_attn=True,
    dataloader_num_workers=4,
    # deepspeed_config_path='./zero2.json',
    # learning_rate=2e-4, # sqrt(16) * 1e-4
    # gradient_accumulation_steps=128,
    system='',
    # dataset=[DatasetName.leetcode_python_en],
    custom_train_dataset_path=co.train_data_path,
    output_dir=co.output_dir,
    gradient_checkpointing=True)

best_ckpt_dir = sft_main(sft_args)
print(f'best_ckpt_dir: {best_ckpt_dir}')