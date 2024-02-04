
train_data_path = ['/tf/user/lgq/data/black_start_case14_versionB.jsonl',
  '/tf/user/lgq/data/black_start_case14_versionB.jsonl',
  '/tf/data/train_data/economic_dispatch_case30_versionA.jsonl',
  '/tf/data/train_data/economic_dispatch_case57_versionA.jsonl',
    '/tf/data/train_data/operation_monitoring_case118_versionA.jsonl',
    '/tf/data/train_data/operation_monitoring_case14_versionA.jsonl',
    '/tf/data/train_data/operation_monitoring_case30_versionA.jsonl',
    '/tf/data/train_data/operation_monitoring_case57_versionA.jsonl',
    '/tf/data/train_data/black_start_case30_versionB.jsonl',
    '/tf/data/train_data/black_start_case30_versionA.jsonl',
    '/tf/data/train_data/black_start_case14_versionB.jsonl',
    '/tf/data/train_data/black_start_case57_versionA.jsonl',
    '/tf/data/train_data/black_start_case57_versionB.jsonl',
    '/tf/data/train_data/black_start_case14_versionA.jsonl',
    '/tf/data/train_data/textbook_instruction_1124.jsonl',
    '/tf/data/train_data/textbook_instruction_augment_1119.jsonl']
val_data_path = '/tf/user/lgq/data/black_start_case14_versionB.jsonl'
# model save path
output_dir = './outputs-ptuning'
# 1 : p-tuning,  2 : adapter
train_type = 2
base_model_path = '/tf/model/Llama-2-7b-chat-ms/'



