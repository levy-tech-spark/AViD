from dataclasses import dataclass
from typing import Dict, Optional, Any
import yaml


@dataclass
class DataConfig:
    train_dir: str
    train_ann: str
    val_dir: str
    val_ann: str
    num_workers: int = 8
    batch_size: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        return cls(
            train_dir=str(data['train_dir']),
            train_ann=str(data['train_ann']),
            val_dir=str(data['val_dir']),
            val_ann=str(data['val_ann']),
            num_workers=int(data.get('num_workers', 8)),
            batch_size=int(data.get('batch_size', 4))
        )

@dataclass
class ModelConfig:
    config_path: str
    weights_path: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            config_path=str(data['config_path']),
            weights_path=str(data['weights_path'])
        )

@dataclass
class TrainingConfig:
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    save_dir: str = 'weights'
    save_frequency: int = 100
    warmup_epochs: int = 5
    use_lora: bool = False
    visualization_frequency: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        return cls(
            num_epochs=int(data.get('num_epochs', 1000)),
            learning_rate=float(data.get('learning_rate', 1e-3)),
            save_dir=str(data.get('save_dir', 'weights')),
            save_frequency=int(data.get('save_frequency', 100)),
            warmup_epochs=int(data.get('warmup_epochs', 5)),
            use_lora=bool(data.get('use_lora', False)),
            visualization_frequency=int(data.get('visualization_frequency', 5))
        )

class ConfigurationManager:
    @staticmethod
    def load_config(config_path: str) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
        """Load configuration from YAML file with type conversion"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        try:
            data_config = DataConfig.from_dict(config['data'])
            model_config = ModelConfig.from_dict(config['model'])
            training_config = TrainingConfig.from_dict(config['training'])
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid configuration format: {str(e)}")
        
        return data_config, model_config, training_config

