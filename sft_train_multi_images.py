from transformers import (
    PreTrainedModel, PretrainedConfig,
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from PIL import Image
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Any
from model import VLMConfig, VLM


def get_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    end_index = 0
    assitant_id = tokenizer('assistant')['input_ids'][0]
    end_id = tokenizer('<|im_end|>')['input_ids'][0]
    while start_index <= len(target)-1:
        if target[start_index]!=assitant_id:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==end_id:
                result.append((start_index+1,end_index+1))
                start_index=end_index+1
    return result

class SftMultiImageDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.image_size = config.image_size
        
        self.datas = pd.read_parquet(self.data_path)
        # id, images: list, conversation: list, source
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas.iloc[index]
        try:
            images = sample['images']
            image_paths = [image_info['path'] for image_info in images]
            conversations = list(sample['conversation'])
            messages = [{"role":"system", "content":'You are a helpful assistant.'}]
            messages.extend(conversations)
            # text = self.tokenizer.apply_chat_template(messages, \
            #     tokenize=False, \
            #     ).replace('<image>', '<|vision_start|>' + '<|image_pad|>'*self.config.image_pad_num + '<|vision_end|>')
            text = self.tokenizer.apply_chat_template(messages, \
                tokenize=False, \
                ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num + '\n')
            input_ids = self.tokenizer(text)['input_ids']
            indexs = get_assistant_tokens(self.tokenizer, input_ids)

            # 填充assistant回答
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for (a_start, a_end) in indexs:
                labels[a_start:a_end] = input_ids[a_start:a_end]
            input_ids = input_ids[:-1] # BOS ->
            labels = labels[1:] # -> EOS
        
            images = []
            for image_name in image_paths:
                image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB').resize(self.image_size)
                images.append(image)
            
            pixel_values = self.processor(text=None, images=images)['pixel_values']

        except:
            
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            # q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"第二张图片的内容是什么\n<image>"}], \
            #     tokenize=False, \
            #     add_generation_prompt=True).replace('<image>', '<|vision_start|>' + '<|image_pad|>'*self.config.image_pad_num + '<|vision_end|>')
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"Describe the second image.\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num + '\n')
            a_text = 'The image is empty.' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


def get_specific_layer_names(model, return_list=True):
    # Create a list to store the layer names
    layer_names = []
    target_patterns = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力机制
        "gate_proj", "up_proj", "down_proj",      # FFN层
    ]
    for name, module in model.named_modules():
        # 只选择语言模型中的层，排除视觉模型和自定义模块
        if "vision" not in name:
            # 检查是否是目标模式
            if any(pattern in name for pattern in target_patterns):
                layer_names.append(name)
    
    # # Recursively visit all modules and submodules
    # for name, module in model.named_modules():
    #     if "vision" not in name and "lm_head" not in name:
    #         # Check if the module is an instance of the specified layers
    #         if isinstance(module, (nn.Linear, nn.Embedding)):
    #             # model name parsing 
    #             layer_names.append(name)
    if return_list:
        return layer_names
    else:
        return set(layer_names)


if __name__ == '__main__':
    config = VLMConfig(
        llm_model_path=r"E:\Production\models\hgf\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
        vision_model_path=r"E:\Production\models\hgf\hub\models--google--siglip-base-patch16-224\snapshots\7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        image_pad_num=49,
        freeze_language_model=False
    )
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained('./output').cuda()

    # LORA 配置
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    target_modules = get_specific_layer_names(model, return_list=True)
    print(target_modules)
    
    # TODO: AdaLoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=16,  # Lora alaph, 类似学习率影响训练效率
        lora_dropout=0.05,  # Dropout
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters()) + 0.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + 0.0
    print(f'模型参数量为：{total_params}') 
    print(f'训练参数量占比：{trainable_params/total_params:.2%}') 

    # Mantis
    images_path = r"E:\Production\models\hgf\hub\datasets--TIGER-Lab--Mantis-Instruct\snapshots\01a9edfe0bb8c2582431308c5b2645a9f4796939\multi_vqa\train_images"
    data_path = r"E:\Production\models\hgf\hub\datasets--TIGER-Lab--Mantis-Instruct\snapshots\01a9edfe0bb8c2582431308c5b2645a9f4796939\multi_vqa\train-00000-of-00001.parquet"
    output_dir = "./sft_output"

    model.train()
    args = TrainingArguments(
        num_train_epochs=2,
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,

        learning_rate=2e-4,
        max_grad_norm=15.0,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        save_steps=200,
        save_strategy="steps",
        save_total_limit=5,
        max_steps=200,
        
        logging_steps=1,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SftMultiImageDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    print(f"Training dataset size: {len(trainer.train_dataset)}")
    print(f"Baseline CrossEntropyLoss(random prediction) ≈ log(vocab_size) = {torch.log(torch.tensor(tokenizer.vocab_size)).item()}")
    # 如果output_dir存在任何checkpoint开头的文件夹, 比如checkpoint-200, 则从该checkpoint开始训练
    # if any("checkpoint" in item and os.path.isdir(os.path.join(output_dir, item)) for item in os.listdir(output_dir)):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()