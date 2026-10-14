import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = "config/logging.yaml") -> logging.Logger:
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Erro letal: Arquivo de configuração de log '{path}' não encontrado.")
        
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Garante que os diretórios dos handlers de arquivo existam
    handlers = config.get("handlers", {})
    for handler in handlers.values():
        if isinstance(handler, dict) and "filename" in handler:
            Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
        
    # Aplica a configuração globalmente
    logging.config.dictConfig(config)
    return logging.getLogger("main")

logger = logging.getLogger("main")