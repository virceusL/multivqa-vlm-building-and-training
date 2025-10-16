from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Dict, Any, Optional, Union

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,llm_model_path = 'Qwen/Qwen2.5-0.5B-Instruct',
                vision_model_path = 'googles/siglip-base-patch16-224',
                image_size = (224, 224),
                freeze_vision_model = True,
                freeze_language_model = True,
                image_pad_num = 49,
                max_num_images = 10,
                use_cross_attn_fusion = True,
                use_qk_norm = True,
                **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.image_size = image_size
        self.freeze_vision_model = freeze_vision_model
        self.freeze_language_model = freeze_language_model
        self.image_pad_num = image_pad_num
        self.max_num_images = max_num_images
        self.use_cross_attn_fusion = use_cross_attn_fusion
        self.use_qk_norm = use_qk_norm
        super().__init__(**kwargs)


def apply_2d_rope(q_or_k, H, W):
    """
    严格的 2D-RoPE，作用在 Q 或 K 上
    q_or_k: [B, seq_len, D], seq_len = H*W
    返回同样 shape 的 tensor
    """
    B, L, D = q_or_k.shape
    assert L == H * W, "seq_len 必须等于 H*W"
    device = q_or_k.device
    dtype = q_or_k.dtype  # 保持数据类型一致
    
    x = q_or_k.view(B, H, W, D)
    half_d = D // 2
    d_h = half_d // 2
    d_w = half_d - d_h
    
    # 使用更稳定的theta计算
    inv_freq_h = 1.0 / (10000 ** (torch.arange(0, d_h, 2, device=device, dtype=dtype) / d_h))
    inv_freq_w = 1.0 / (10000 ** (torch.arange(0, d_w, 2, device=device, dtype=dtype) / d_w))
    
    pos_h = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
    pos_w = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)
    
    theta_h = pos_h * inv_freq_h
    theta_w = pos_w * inv_freq_w
    
    # 添加数值稳定性检查
    theta_h = torch.clamp(theta_h, min=-50, max=50)
    theta_w = torch.clamp(theta_w, min=-50, max=50)
    
    sin_h, cos_h = theta_h.sin(), theta_h.cos()
    sin_w, cos_w = theta_w.sin(), theta_w.cos()

    # 对高度维度旋转前 d_h
    x_h = x[..., :d_h]  # [B,H,W,d_h]
    x_h_even = x_h[..., 0::2]
    x_h_odd  = x_h[..., 1::2]
    x_h_rot = torch.stack([x_h_even * cos_h - x_h_odd * sin_h,
                           x_h_even * sin_h + x_h_odd * cos_h], dim=-1)
    x_h_rot = x_h_rot.flatten(-2)

    # 对宽度维度旋转剩余 d_w
    x_w = x[..., d_h: d_h+d_w]  # [B,H,W,d_w]
    x_w_even = x_w[..., 0::2]
    x_w_odd  = x_w[..., 1::2]
    x_w_rot = torch.stack([x_w_even * cos_w - x_w_odd * sin_w,
                           x_w_even * sin_w + x_w_odd * cos_w], dim=-1)
    x_w_rot = x_w_rot.flatten(-2)

    # 余下维度直接保留
    x_rest = x[..., d_h+d_w:]  # [B,H,W,D-(d_h+d_w)]

    # 拼回
    x_rot = torch.cat([x_h_rot, x_w_rot, x_rest], dim=-1)
    return x_rot.view(B, L, D)


class SingleLayerAdapter2DRoPE(nn.Module):
    def __init__(self, hidden_dim, num_queries=49, num_heads=8, dropout=0.1, use_qk_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.use_qk_norm = use_qk_norm

        # 可学习 query
        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(hidden_dim)
            self.k_norm = nn.RMSNorm(hidden_dim)


        # LayerNorm
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # MLP
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 4, hidden_dim),
        #     nn.Dropout(dropout)
        # )

    def forward(self, image_features, H, W):
        """
        Hybrid Normalization Strategy: QK-Norm + Post-LN
        Args:
            image_features: [B, image_seq_len, hidden_dim]  # KV
            H: patch grid height
            W: patch grid width
        """
        B = image_features.size(0)
        queries = self.query_embeddings.repeat(B, 1, 1)
        # shape (B, num_queries, hidden_dim)

        # qk_norm
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            image_features = self.k_norm(image_features)

        # 2D-QK-RoPE
        # q_rot = queries
        q_rot = apply_2d_rope(queries, 1, self.num_queries)  # queries 1xnum_queries
        k_rot = apply_2d_rope(image_features, H, W)  # keys: HxW grid

        # Cross Attention
        attn_output, _ = self.cross_attn(query=q_rot, key=k_rot, value=image_features)

        # Residual & Post-LN
        queries = queries + attn_output
        queries = self.post_attn_norm(queries)

        # # MLP
        # mlp_out = self.mlp(queries)
        # queries = queries + mlp_out

        return queries  # [B, num_queries, hidden_dim]


# 用户输入的图像顺序的位置编码
class ImageOrderEmbedding(nn.Module):
    """
    改进的图像顺序编码模块，支持更灵活的多图场景
    """
    def __init__(self, hidden_dim, max_num_images=10, dropout=0.1):
        super().__init__()
        self.max_num_images = max_num_images
        self.hidden_dim = hidden_dim
        
        # 位置嵌入，支持0到max_num_images-1的位置
        self.order_emb = nn.Embedding(max_num_images, hidden_dim)
        
        # 可选的dropout和layer norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        # 使用较小的标准差初始化，避免影响原始特征太多
        nn.init.normal_(self.order_emb.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                image_features: torch.Tensor, 
                image_positions: Optional[torch.Tensor] = None,
                normalize: bool = True) -> torch.Tensor:
        """
        Args:
            image_features: [B_img, patch_num, hidden_dim] 图像特征
            image_positions: [B_img] 可选的图像位置索引，如果为None则使用默认顺序
            normalize: 是否应用layer normalization
        Returns:
            带位置编码的图像特征: [B_img, patch_num, hidden_dim]
        """
        B_img, patch_num, feature_dim = image_features.shape
        device = image_features.device
        
        # 检查维度
        if feature_dim != self.hidden_dim:
            raise ValueError(f"特征维度 {feature_dim} 与编码维度 {self.hidden_dim} 不匹配")
        
        # 生成位置索引
        if image_positions is None:
            # 默认使用顺序索引 0, 1, 2, ...
            if B_img > self.max_num_images:
                raise ValueError(f"图像数量 {B_img} 超过最大支持数量 {self.max_num_images}")
            image_positions = torch.arange(B_img, device=device)
        else:
            # 验证提供的位置索引
            if image_positions.max() >= self.max_num_images:
                raise ValueError(f"位置索引超出范围 [0, {self.max_num_images-1}]")
            if len(image_positions) != B_img:
                raise ValueError(f"位置索引数量 {len(image_positions)} 与图像数量 {B_img} 不匹配")
        
        # 获取位置嵌入 [B_img, hidden_dim]
        order_vectors = self.order_emb(image_positions)
        
        # 广播到所有patch [B_img, patch_num, hidden_dim]
        order_vectors = order_vectors.unsqueeze(1).expand(B_img, patch_num, feature_dim)
        
        # 添加位置编码
        enhanced_features = image_features + order_vectors
        
        # 可选的normalization和dropout
        if normalize:
            enhanced_features = self.layer_norm(enhanced_features)
        
        if self.dropout is not None:
            enhanced_features = self.dropout(enhanced_features)
            
        return enhanced_features


class MultiImageProcessor(nn.Module):
    """
    多图像处理器，处理复杂的多图场景
    """
    def __init__(self, hidden_dim, max_num_images=10, image_pad_num=49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_num_images = max_num_images
        self.image_pad_num = image_pad_num
        
        self.image_order_emb = ImageOrderEmbedding(
            hidden_dim=hidden_dim, 
            max_num_images=max_num_images,
            dropout=0.1
        )
    
    def process_images(self, 
                      image_features: torch.Tensor,
                      images_per_sample: Union[int, List[int]],
                      image_positions: Optional[List[int]] = None) -> torch.Tensor:
        """
        处理多图像输入，支持不同样本有不同数量的图像
        
        Args:
            image_features: [B_img, patch_num, hidden_dim] 所有图像特征
            images_per_sample: 每个样本的图像数量，可以是int(所有样本相同)或List[int]
            image_positions: 可选的图像位置列表
        
        Returns:
            processed_features: [B_text, max_images*patch_num, hidden_dim]
        """
        B_img, patch_num, hidden_dim = image_features.shape
        
        # 标准化images_per_sample
        if isinstance(images_per_sample, int):
            # 所有样本都有相同数量的图像
            B_text = B_img // images_per_sample
            images_per_sample = [images_per_sample] * B_text
        else:
            B_text = len(images_per_sample)
            if sum(images_per_sample) != B_img:
                raise ValueError(f"图像总数 {B_img} 与指定分配 {sum(images_per_sample)} 不匹配")
        
        # 应用位置编码
        if image_positions is not None:
            image_positions_tensor = torch.tensor(image_positions, device=image_features.device)
            enhanced_features = self.image_order_emb(image_features, image_positions_tensor)
        else:
            enhanced_features = self.image_order_emb(image_features)
        
        # 重组为每个文本样本的图像组
        max_images = max(images_per_sample)
        result_features = torch.zeros(
            B_text, max_images * patch_num, hidden_dim,
            device=image_features.device, dtype=image_features.dtype
        )
        
        img_idx = 0
        for text_idx, num_imgs in enumerate(images_per_sample):
            if num_imgs > 0:
                # 获取当前样本的图像特征
                sample_features = enhanced_features[img_idx:img_idx + num_imgs]  # [num_imgs, patch_num, hidden_dim]
                # 展平并填入结果
                flattened = sample_features.view(-1, hidden_dim)  # [num_imgs*patch_num, hidden_dim]
                result_features[text_idx, :len(flattened)] = flattened
                img_idx += num_imgs
        
        return result_features
    
    def get_attention_mask(self, 
                          images_per_sample: Union[int, List[int]], 
                          B_text: int) -> torch.Tensor:
        """
        生成图像部分的attention mask
        
        Returns:
            mask: [B_text, max_images*patch_num] 1表示有效位置，0表示padding
        """
        if isinstance(images_per_sample, int):
            images_per_sample = [images_per_sample] * B_text
        
        max_images = max(images_per_sample)
        mask = torch.zeros(B_text, max_images * self.image_pad_num)
        
        for text_idx, num_imgs in enumerate(images_per_sample):
            if num_imgs > 0:
                mask[text_idx, :num_imgs * self.image_pad_num] = 1
        
        return mask


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_qk_norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(hidden_size)
            self.k_norm = nn.RMSNorm(hidden_size)
        
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, text_embeddings, image_features, attention_mask=None):
        """
        Hybrid Normalization: QK-Norm + Post-LN
        Args:
            text_embeddings: [batch_size, text_seq_len, hidden_size]
            image_features: [batch_size, image_seq_len, hidden_size]
            attention_mask: [batch_size, text_seq_len]
        """
        if self.use_qk_norm:
            text_embeddings = self.q_norm(text_embeddings)
            image_features = self.k_norm(image_features)
        # Cross Attention
        attn_output, _ = self.cross_attention(
            query=text_embeddings,
            key=image_features,
            value=image_features,
            key_padding_mask=None
        )
        text_embeddings = text_embeddings + attn_output
        text_embeddings = self.post_attn_norm(text_embeddings)
        
        # MLP (PreLN)
        mlp_out = self.mlp(text_embeddings)
        text_embeddings = text_embeddings + mlp_out

        return text_embeddings


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)

        vm_dim = self.vision_model.config.vision_config.hidden_size
        llm_dim = self.llm_model.config.hidden_size

        # Adapter Layer - Compresses vision model features to a smaller set of queries (learnable)
        # Q: [1(B), 49, vm_dim], KV: [B, img_seq, vm_dim] --> [B, 49, vm_dim]
        self.adapter = SingleLayerAdapter2DRoPE(hidden_dim=vm_dim, num_queries=self.config.image_pad_num, use_qk_norm=self.config.use_qk_norm)

        # Linear Projector - Maps vision model features to LLM hidden size
        self.linear_projector = nn.Sequential(
            nn.Linear(vm_dim, llm_dim), # 768 -->  896
            nn.RMSNorm(llm_dim, eps=1e-6),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )

        # Image Ordering Embedding - Adds positional information to image features
        # to help the model distinguish between different images in a batch (especially when multiple images are associated with a single text input)
        self.multi_image_processor = MultiImageProcessor(hidden_dim=llm_dim, max_num_images=self.config.max_num_images, image_pad_num=self.config.image_pad_num)
        
        # Cross-Attention Fusion Layer - Combines text and image features
        if self.config.use_cross_attn_fusion:
            self.cross_attention_fusion = CrossAttentionFusion(hidden_size=llm_dim, use_qk_norm=self.config.use_qk_norm)

        # Freeze vision model parameters to retain their pre-trained generalization capabilities
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Freeze language model parameters to prevent them from being updated since we want to train the fusion layers first
        if self.config.freeze_language_model:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        self.image_pad_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, 
                input_ids, 
                labels, 
                pixel_values, 
                attention_mask=None, 
                inputs_embeds=None, 
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                images_per_sample=None, 
                image_positions=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 

        B_text, seq_len, llm_dim = text_embeds.shape # (B_text, seq, 896) llm_dim=896
        B_img, patches, vm_dim = image_embeds.shape # (B_img, 196, 768) vm_dim=768

        # 图像特征压缩与映射
        # 使用 Adapter 将原始视觉特征 (14*14=196) 压缩到 image_pad_num 个可学习 token，
        # 并通过 Feature Mapping 映射到 LLM hidden size
        adapter_output = self.adapter(image_embeds, H=14, W=14) # (B_img, 196, 768) --> (B_img, 49, 768)
        image_features = self.linear_projector(adapter_output) # (B_img, 49, 768) --> (B_img, 49, 896)

        # 多图拼接
        if B_img > 1:
            if B_text == 1:
                image_features = self.multi_image_processor.process_images(image_features=image_features, images_per_sample=B_img)
            elif B_img % B_text == 0:
                image_features = self.multi_image_processor.process_images(image_features=image_features, images_per_sample=B_img//B_text)
            elif images_per_sample:
                image_features = self.multi_image_processor.process_images(image_features=image_features, images_per_sample=images_per_sample, image_positions=image_positions)
            else:
                raise ValueError(f"Incompatible image and text batch sizes with wrong image_per_sample and image_positions: B_img={B_img}, B_text={B_text}, image_per_sample={images_per_sample}, image_positions={image_positions}")
        elif B_text > 1:
            raise ValueError(f"Incompatible image and text batch sizes: B_img={B_img}, B_text={B_text}")
        
        text_embeds = text_embeds.to(image_features.dtype)
        
        # 将图像特征和文本特征融合
        # 如果使用交叉注意力，则图像特征作为键值对，文本特征作为查询进行融合
        if self.config.use_cross_attn_fusion:
            # text_embeds: (B_text, seq, 896) (Query); image_features: (B_img, 49, 896) (Key/Value)
            inputs_embeds = self.cross_attention_fusion(text_embeds, image_features, attention_mask)
        else:
            mask = (input_ids == self.image_pad_id)
            batch_indices, image_indices = torch.where(mask)
            num_slots = batch_indices.numel()
            expected = image_features.shape[0] * image_features.shape[1]
            if num_slots != expected:
                raise ValueError(f"<|image_pad|> numbers ({num_slots}) and image_features ({expected}) unmatched")
            inputs_embeds[batch_indices, image_indices] = image_features.reshape(-1, llm_dim)
        
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def get_input_embeddings(self):
        """返回模型的输入嵌入层"""
        return self.llm_model.get_input_embeddings()

    def get_output_embeddings(self):
        """返回模型的输出嵌入层"""
        # 如果你的模型有专门的输出嵌入层，返回它
        # 否则，通常返回与输入嵌入层相同的层
        return self.llm_model.get_output_embeddings() if hasattr(self.llm_model, 'get_output_embeddings') else self.llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """设置模型的输入嵌入层"""
        self.llm_model.set_input_embeddings(value)