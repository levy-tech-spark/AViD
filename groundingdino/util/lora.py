import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


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

def load_lora_weights(model, weights_path):
    lora_state_dict = torch.load(weights_path)
    model.load_state_dict(lora_state_dict, strict=False)


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


def add_lora_to_model(model, rank=8):
    """
    Adds LoRA to linear layers in Grounding DINO model
    """
    # List all linear layer names from model architecture
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=[
            # Transformer Encoder layers
            "sampling_offsets",  # MultiScaleDeformableAttention
            "attention_weights",
            "value_proj",
            "output_proj",
            "linear1",  # FFN
            "linear2",
            "out_proj",  # MultiheadAttention
            # Transformer Decoder layers
            "sampling_offsets",  # MultiScaleDeformableAttention
            "attention_weights",
            "value_proj",
            "output_proj",
            "linear1",  # FFN
            "linear2",
            # BiMultiHeadAttention layers
            "v_proj",
            "l_proj",
            "values_v_proj",
            "values_l_proj",
            "out_v_proj",
            "out_l_proj",
            # Query, Key, Value projections
            "q_proj",
            "k_proj",
            "v_proj",
            # Various other linear layers
            "feat_map",
            "query",
            "key",
            "value",
            "dense",    
        ],
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA to entire model
    print("Adding LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    # Freeze base model
    print("\nFreezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    print("Unfreezing LoRA parameters...")
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            print(f"Unfroze: {n}")

    # Log stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA Trainable params: {trainable_params} ({100 * trainable_params / total_params:.2f}% of total)")
    
    return model

def add_lora_to_layers(model, rank=32):
    """
    Adds LoRA to linear layers in Grounding DINO model
    """

    lora_mlp_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["0", "1", "2"],  # MLP layer indices
        #lora_dropout=1e-3,
        bias="none",
    )

    print("\nAdding LoRA to components:")
    
    #Add LoRA to bbox prediction MLPs 
    for i in range(len(model.bbox_embed)):
        model.bbox_embed[i] = get_peft_model(model.bbox_embed[i], lora_mlp_config)
    print("✓ Added to bbox prediction")


    # Configuration for decoder layers based on names
    decoder_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=[
            # Cross attention components
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
            "linear2"
        ],
        #lora_dropout=0.1,
        bias="none",
    )

    # Add LoRA to decoder layers
    for i, layer in enumerate(model.transformer.decoder.layers):
        try:
            model.transformer.decoder.layers[i] = get_peft_model(layer, decoder_config)
            print(f"✓ Added to decoder layer {i}")
        except Exception as e:
            print(f"Failed to add LoRA to decoder layer {i}: {str(e)}")

    ## Add lora to feature projection layer
    ## Since it has no name we have to do this step additionally
    feat_map_module = nn.ModuleDict({"feat_map": model.feat_map}) 

    feat_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["feat_map"],  # Now we can target by this name
        #lora_dropout=0.1,
        bias="none",
    )

    # 3. Add LoRA to text feature projection
    try:
        # Apply LoRA to wrapped version and extract back
        lora_feat = get_peft_model(feat_map_module, feat_config)
        model.feat_map = lora_feat.feat_map  # Extract the layer back
        print("✓ Added to text feature projection")
    except Exception as e:
        print(f"Failed to add LoRA to text feature projection: {str(e)}")

    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only LoRA parameters
    for n, p in model.named_parameters():
        if 'lora_' in n:
            print(f"Unfrooze {n} ")
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA parameters: {trainable_params:,} / {total_params:,} = {100 * trainable_params / total_params:.2f}%")

    return model



def add_lora_to_layers2(model, rank=32):
    """
    Adds LoRA to complete Grounding DINO model using PEFT's functionality
    """
    from peft import LoraConfig, get_peft_model

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

    print("\nAll Model Layers:")
    for name, module in model.named_modules():
        if 'bbox' in name:
            print(f"full path: {name}")
    

    print("Converting model to LoRA...")
    model = get_peft_model(model, config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA parameters: {trainable_params:,} / {total_params:,} = {100 * trainable_params / total_params:.2f}%")

    return model

