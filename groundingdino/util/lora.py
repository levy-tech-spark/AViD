import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def get_lora_weights(model):
    ## This needs lora config which is not part of grouding dino model state dict tight now hack to get lora weights
    #lora_state_dict = get_peft_model_state_dict(model)
    # Collect LoRA parameters manually
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data.cpu()
    # If no LoRA weights found, print warning
    if not lora_state_dict:
        print("No LoRA weights found in the model.")
    return lora_state_dict

def get_lora_optimizer_params(model):
    # Get only LoRA trainable parameters for optimizer
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)
    return lora_params

def verify_only_lora_trainable(model):
    """
    Verifies that only LoRA parameters are trainable and counts parameters accurately
    """
    trainable_non_lora = []
    lora_params = 0
    trainable_lora = 0
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params += param.numel()
            if param.requires_grad:
                trainable_lora += param.numel()
        elif param.requires_grad:
            trainable_non_lora.append(name)
    
    if trainable_non_lora:
        print("WARNING: Found non-LoRA trainable parameters:")
        for name in trainable_non_lora:
            print(f"- {name}")
        return False
    
    print(f"✓ Only LoRA parameters are trainable")
    print(f"Total LoRA parameters: {lora_params:,}")
    print(f"Trainable LoRA parameters: {trainable_lora:,}")
    return True

def add_lora_to_model(model, rank=32):
    """
    Adds LoRA to complete Grounding DINO model using PEFT's functionality"""

    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=[
            # Decoder cross attention
            "cross_attn.sampling_offsets",
            "cross_attn.attention_weights", 
            "cross_attn.value_proj",
            "cross_attn.output_proj",
            # Text cross attention
            "ca_text.out_proj",
            # Self attention 
            "self_attn.out_proj",
            # FFN
            "linear1",
            "linear2",
            # Bbox prediction layers
            "bbox_embed.0.layers.0",
            "bbox_embed.0.layers.1",
            # fearue map
            "feat_map"
        ],
        modules_to_save=["bbox_embed.0.layers.2"],
        bias="none",
        inference_mode=False,
    )
    #print("\nAll Model Layers:")
    #for name, module in model.named_modules():
    #    if 'bbox' in name:
    #        print(f"full path: {name}")
    print("Converting model to LoRA...")
    model = get_peft_model(model, config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA parameters: {trainable_params:,} / {total_params:,} = {100 * trainable_params / total_params:.2f}%")

    return model

