import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

class DialogueDataset(Dataset):
    def __init__(se -blf, file_path, tokenizer, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.examples = []
        for line in lines:
            self.examples.append(tokenizer.encode(line, add_special_tokens=True))

        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', force_download=True)
model = GPT2LMHeadModel.from_pretrained('gpt2', force_download=True)

# 加载数据集
dataset = DialogueDataset('dialogues.txt', tokenizer)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    DialogueDataset()
