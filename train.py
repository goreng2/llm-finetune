from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        # text = f"### Instruct: {example['instruction'][i]}\n" + \
        #         f"### Input: {example['input'][i]}\n" + \
        #         f"### Output: {example['output'][i]}\n"
        text = f"아래는 감성대화를 위한 대화쌍입니다." + "\n\n"\
                f"### 입력:\n{example['prompt'][i]}" + "\n\n"\
                f"### 응답:\n{example['prompt'][i]}" + "\n"
        output_texts.append(text)
    return output_texts


# Dataset
# dataset_name = 'nlpai-lab/kullm-v2'  # https://huggingface.co/datasets/nlpai-lab/kullm-v2
# dataset = load_dataset(dataset_name, split="train")
# dataset = load_dataset("json", data_files="dataset/nl2sql/nl2sql.json", split="train")
dataset = load_dataset("json", data_files="dataset/emotion/emotion.json", split="train")
# print(dataset)

# Loading the model
model_name = "beomi/llama-2-ko-7b"  # https://huggingface.co/beomi/llama-2-ko-7b
# model_name = "beomi/llama-2-ko-70b"  # https://huggingface.co/beomi/llama-2-ko-70b
# model_name = "hyunseoki/ko-en-llama2-13b"  # https://huggingface.co/hyunseoki/ko-en-llama2-13b

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

# Loading the trainer
output_dir = "results_llama2-7b_emotion2"
per_device_train_batch_size = 128
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 10000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

max_seq_length = 128

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    # dataset_text_field="text",
    # packing=True,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train the model
trainer.train()
# trainer.train(resume_from_checkpoint="results_llama2-13b-enko_nl2sql/checkpoint-3600")

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")

# lora_config = LoraConfig.from_pretrained('outputs')
# model = get_peft_model(model, lora_config)

# device = "cuda:0"

# text = "1인용 매트리스의 수수료를 찾아줘"
# text = f"### Instruct: 다음 문장을 SQL 쿼리문으로 바꿔주세요.\n" + \
#         f"### Input: 수수료가 10000 이상인 폐기물의 품명과 규격을 알려줘\n" + \
#         f"### Output: "
# inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
