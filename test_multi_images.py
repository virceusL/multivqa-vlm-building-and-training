from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM

device = "cuda"
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained('./sft_output')
model.to(device)
# q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":'第二张图上面有什么\n<image><image>'}], \
#             tokenize=False, \
#             add_generation_prompt=True).replace('<image>', '<|vision_start|>' + '<|image_pad|>'*49 + '<|vision_end|>')

q_text = tokenizer.apply_chat_template(
    [
        {"role":"system", "content":'You are a helpful assistant.'}, 
        {"role":"user", "content":'<image><image>\nIf someone wants to sit and read books, what would be the advantages of image 1 and image 2 and why?'}
    ], 
    tokenize=False, 
    add_generation_prompt=True
).replace('<image>', '<|image_pad|>'*49 + '\n')

input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
input_ids = input_ids.to(device)
image1 = Image.open('./test_1.jpg').convert("RGB")
image2 = Image.open('./test_2.jpg').convert("RGB")
pixel_values = processor(text=None, images=[image1, image2]).pixel_values
pixel_values = pixel_values.to(device)
model.eval()
import torch
from torch.nn import functional as F
max_new_tokens = 100
temperature = 0.0
eos = tokenizer.eos_token_id
top_k = None
s = input_ids.shape[1]
while input_ids.shape[1] < s + max_new_tokens - 1:  
    inference_res = model(input_ids, None, pixel_values)  
    logits = inference_res.logits 
    logits = logits[:, -1, :] 

    for token in set(input_ids.tolist()[0]):  
        logits[:, token] /= 1.0

    if temperature == 0.0: 
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature  
        if top_k is not None:  
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf') 

        probs = F.softmax(logits, dim=-1)  
        idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

    if idx_next == eos:  
        break

    input_ids = torch.cat((input_ids, idx_next), dim=1)  
print(tokenizer.decode(input_ids[:, s:][0]))