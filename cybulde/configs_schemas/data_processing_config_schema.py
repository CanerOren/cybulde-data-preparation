from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from omegaconf import MISSING
from dataclasses import field

from cybulde.configs_schemas.infrastructure import gcp_schema
from cybulde.configs_schemas.data_processing import dataset_readers_schema



@dataclass 
class DataProcessingConfig:
    version: str = MISSING
    data_local_save_dir: str = "./data/raw"
    dvc_remote_repo: str = "https://github.com/CanerOren/cybulde-data.git"
    dvc_data_folder: str = "data/raw"
    github_user_name: str = "CanerOren"
    github_acces_token_secret_id: str = "cybulde-data-github-access-token"

    infrastructure: gcp_schema.GCPConfig = field(default_factory=gcp_schema.GCPConfig)

    dataset_reader_manager: dataset_readers_schema.DatasetReaderManagerConfig = MISSING

def setup_config() -> None:
    gcp_schema.setup_config()
    dataset_readers_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="data_processing_config_schema", node=DataProcessingConfig)