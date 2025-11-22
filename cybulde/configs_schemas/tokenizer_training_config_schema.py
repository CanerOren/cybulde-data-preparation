from dataclasses import field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from cybulde.configs_schemas.infrastructure import gcp_schema
from cybulde.configs_schemas.tokenization import tokenizer_schema


@dataclass
class TokenizerTrainingConfig:
    infrastructure: gcp_schema.GCPConfig = field(default_factory=gcp_schema.GCPConfig)

    data_parquet_path: str = MISSING
    text_column_name: str = MISSING

    tokenizer: tokenizer_schema.TokenizerConfig = MISSING

    docker_image_name: str = MISSING
    docker_image_tag: str = MISSING


def setup_config() -> None:
    gcp_schema.setup_config()
    tokenizer_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="tokenizer_training_config_schema", node=TokenizerTrainingConfig)
