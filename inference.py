import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Loading the model
# model_name = "beomi/llama-2-ko-7b"  # https://huggingface.co/beomi/llama-2-ko-7b
model_name = "kfkas/Llama-2-ko-7b-Chat"  # https://huggingface.co/kfkas/Llama-2-ko-7b-Chat
# model_name = "hyunseoki/ko-en-llama2-13b"  # https://huggingface.co/hyunseoki/ko-en-llama2-13b

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# lora_config = LoraConfig.from_pretrained('outputs')
lora_config = LoraConfig.from_pretrained('results_llama2-7b_emotion2/checkpoint-10000')
model = get_peft_model(model, lora_config)

text = f"아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n" + \
        f"### 명령어:\n감성 대화를 위한 응답을 출력하세요.\n\n" + \
        f"### 입력:\n오늘 미세먼지가 장난 아니구만\n\n" + \
        f"### 응답:\n"
inputs = tokenizer(text, return_tensors="pt").to("cuda:2")
outputs = model.generate(**inputs,
                         max_new_tokens=50,
                         early_stopping=True,
                         do_sample=True,
                         top_k=20,
                         top_p=0.92,
                         no_repeat_ngram_size=3,
                         repetition_penalty=1.2,
                         num_beams=3,
                         eos_token_id=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
