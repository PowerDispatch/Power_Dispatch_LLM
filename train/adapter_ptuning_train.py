import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from modelscope import Model, AutoModelForSequenceClassification, AutoTokenizer, MsDataset
from swift import Swift, LoRAConfig, AdapterConfig, Trainer, TrainingArguments, PromptConfig
import torch
from transformers import default_data_collator
import conf as co


model = Model.from_pretrained(co.base_model_path, torch_dtype=torch.bfloat16)
adapter_config = AdapterConfig(
                dim=model.config.hidden_size,
                target_modules=['mlp'],
                method_name='forward',
                hidden_pos=0,
                adapter_length=32,
            )

prompt_config = PromptConfig(
    dim=768,  # hidden states的维度
    target_modules=r'.*layer\.\d+$',  # 要使用正则表达式替换的模块
    embedding_pos=0,    # embedding张量的位置
    prompt_length=10,   # 提示符token的长度
    attach_front=False  # 是否将提示符附加在embedding前面
)

if co.train_type == 2:
    train_type = adapter_config
else:
    train_type = prompt_config

model = Swift.prepare_model(model, train_type)
tokenizer = AutoTokenizer.from_pretrained('/tf/model/Llama-2-7b-chat-ms/')


train_dataset = MsDataset.load(dataset_name = '/tf/user/lgq/data',data_files={'train': co.train_data_path}).to_hf_dataset()
val_dataset = MsDataset.load(dataset_name = '/tf/user/lgq/data',data_files={'val': co.val_data_path}).to_hf_dataset().select(range(4))


def tokenize_function(examples):
    return tokenizer(examples['instruction'], examples['input'],examples['output'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


arguments = TrainingArguments(
    output_dir= co.output_dir,
    per_device_train_batch_size=1,
)

trainer = Trainer(model, arguments, train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=default_data_collator,)
trainer.train()

