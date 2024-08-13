
import torch

def freeze_model_layers(model, freeze_config=None) -> None:
    """
    Selectively freeze model layers based on configuration.
    Special case: -1 means keep all layers of that component trainable
    """
    if freeze_config is None:
        freeze_config = {
            'backbone': True,
            'text_encoder': True, 
            'projections': True,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'exclude_patterns': ['class_embed', 'bbox_embed']
        }
    
    def should_freeze(name):
        # Check exclude patterns first
        if any(pattern in name for pattern in freeze_config.get('exclude_patterns', [])):
            return False
        return True
    
    def freeze_component(module, component_name):
        print(f"\nFreezing {component_name}")
        for name, param in module.named_parameters():
            if should_freeze(name):
                param.requires_grad = False
                print(f"Froze: {component_name}.{name}")
    
    # Backbone
    if hasattr(model, 'backbone') and freeze_config.get('backbone', True):
        freeze_component(model.backbone, "backbone")
        
    # Text Encoder (BERT)
    if hasattr(model, 'bert') and freeze_config.get('text_encoder', True):
        freeze_component(model.bert, "bert")
            
    # Projections
    if freeze_config.get('projections', True):
        if hasattr(model, 'input_proj'):
            freeze_component(model.input_proj, "input_proj")
        if hasattr(model, 'feat_map'):
            freeze_component(model.feat_map, "feat_map")
                
    # Transformer Encoder Layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
        num_encoder = len(model.transformer.encoder.layers)
        trainable_encoder = freeze_config.get('encoder_layers', 0)
        
        if trainable_encoder != -1:
            # Freeze main encoder layers
            if trainable_encoder < num_encoder:
                for i in range(num_encoder - trainable_encoder):
                    freeze_component(model.transformer.encoder.layers[i], f"encoder.layer_{i}")
            
            # Freeze text layers
            if hasattr(model.transformer.encoder, 'text_layers'):
                for i in range(len(model.transformer.encoder.text_layers)):
                    if i < (num_encoder - trainable_encoder):
                        freeze_component(model.transformer.encoder.text_layers[i], f"encoder.text_layer_{i}")
            
            # Freeze fusion layers
            if hasattr(model.transformer.encoder, 'fusion_layers'):
                for i in range(len(model.transformer.encoder.fusion_layers)):
                    if i < (num_encoder - trainable_encoder):
                        freeze_component(model.transformer.encoder.fusion_layers[i], f"encoder.fusion_layer_{i}")
                
    # Transformer Decoder Layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
        num_decoder = len(model.transformer.decoder.layers)
        trainable_decoder = freeze_config.get('decoder_layers', 1)
        
        if trainable_decoder != -1 and trainable_decoder < num_decoder:
            for i in range(num_decoder - trainable_decoder):
                freeze_component(model.transformer.decoder.layers[i], f"decoder.layer_{i}")



def print_frozen_status(model) -> None:
    """
    Enhanced version that prints component-level details
    """
    frozen_count = 0
    trainable_count = 0
    
    print("\n=== Component Status ===")
    for name, module in model.named_children():
        component_frozen = 0
        component_trainable = 0
        
        print(f"\n{name}:")
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                component_trainable += param.numel()
                print(f"  🔥 {param_name}")
            else:
                component_frozen += param.numel()
                print(f"  ❄️ {param_name}")
                
        frozen_count += component_frozen
        trainable_count += component_trainable
        
        print(f"  Frozen parameters: {component_frozen:,}")
        print(f"  Trainable parameters: {component_trainable:,}")
    
    total_params = frozen_count + trainable_count
    print("\n=== Overall Status ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Frozen Parameters: {frozen_count:,} ({frozen_count/total_params:.2%})")
    print(f"Trainable Parameters: {trainable_count:,} ({trainable_count/total_params:.2%})")

