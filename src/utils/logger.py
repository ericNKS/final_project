import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = "config/logging.yaml") -> None:
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Erro letal: Arquivo de configuração de log '{path}' não encontrado.")
        
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Aplica a configuração globalmente
    logging.config.dictConfig(config)