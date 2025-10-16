from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any, Optional, Union
from model import VLMConfig, VLM

class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config, split='train'):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.image_size = config.image_size

        if 'coco' in self.data_path:
            from datasets import load_dataset
            self.datas = load_dataset(self.data_path)
            self.datas = self.datas[split].shuffle(seed=42).select(range(len(self.datas[split])//8))
        elif 'LLaVA' in self.data_path:
            import random
            # 设置种子
            random.seed(42)
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.datas = json.load(f)   
            # 随机选12.5%
            self.datas = random.sample(self.datas, int(len(self.datas)//8))
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            if 'coco' in self.data_path:
                # coco_captions
                caption = sample['caption']
                image = sample['image'].convert('RGB').resize(self.image_size)
                q_text = self.tokenizer.apply_chat_template(
                    [
                        {"role":"system", "content":'You are a helpful assistant.'},
                        {"role":"user", "content":"<image>\nDescribe what you see in this image."}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)

                a_text = caption + self.tokenizer.eos_token
            
            elif 'LLaVA' in self.data_path:
                # LLaVA-CC3M
                conversations = sample['conversations']
                image_name = sample['image']
                image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB").resize(self.image_size)
                q_text = self.tokenizer.apply_chat_template(
                    [
                        {"role":"system", "content":'You are a helpful assistant.'},
                        {"role":"user", "content":conversations[0]['value']}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)

                a_text = conversations[1]['value'] + self.tokenizer.eos_token

            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids

            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
            
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', self.image_size, color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template(
                [
                    {"role":"system", "content":'You are a helpful assistant.'},
                    {"role":"user", "content":"<image>\nDescribe what you see in this image."}
                ],
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)

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
            
        
        
if __name__ == '__main__':
    config = VLMConfig(
        llm_model_path=r"E:/Production/models/hgf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
        vision_model_path=r"E:/Production/models/hgf/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        image_pad_num=49
    )
    model = VLM(config).cuda()
    total_params = sum(p.numel() for p in model.parameters()) + 0.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + 0.0
    print(f'模型参数量为：{total_params}') 
    print(f'训练参数量占比：{trainable_params/total_params:.2%}') 

    use_llava = True
    if use_llava:
        images_path = r"E:/Production/models/hgf/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590/images"
        data_path = r"E:/Production/models/hgf/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590/chat.json"
        # data_path = r"E:/Production/models/hgf/hub/datasets--LinkSoul--Chinese-LLaVA-Vision-Instructions/snapshots/d10ec6e47aa6b9aa3a53d03d0793eb4728e921f3/LLaVA-CC3M-Pretrain-595K/chat_translated.json"
    else:
        images_path = None
        data_path = "jxie/coco_captions"

    output_dir = './pre_output'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    args = TrainingArguments(
        num_train_epochs=2,
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,

        learning_rate=1e-4,
        max_grad_norm=5.0,
        warmup_steps=1000,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        save_steps=2000,
        save_strategy="steps",
        save_total_limit=5,
        # max_steps=100,
        
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        # eval_dataset=MyDataset(images_path, data_path, tokenizer, processor, config, split='validation'),
        data_collator=MyDataCollator(tokenizer)  
    )
    print(f"Training dataset size: {len(trainer.train_dataset)}")
    print(f"Baseline CrossEntropyLoss(random prediction) ≈ log(vocab_size) = {torch.log(torch.tensor(tokenizer.vocab_size)).item()}")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()
    # trainer.evaluate()