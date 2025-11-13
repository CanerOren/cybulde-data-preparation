# import pandas as pd
from hydra.utils import instantiate
from pathlib import Path
import os

from dask.distributed import Client
import dask.dataframe as dd

from cybulde.configs_schemas.data_processing_config_schema import DataProcessingConfig
from cybulde.utils.config_utils import get_config
from cybulde.utils.data_utils import get_raw_data_with_version
from cybulde.utils.gcp_utils import access_secret_version
from cybulde.utils.utils import get_logger
from cybulde.data_processing.dataset_cleaners import DatasetCleanerManager

def process_raw_data(df_partition: dd.DataFrame, dataset_cleaner_manager: DatasetCleanerManager) -> dd.Series:
    return df_partition["text"].apply(dataset_cleaner_manager)

@get_config(config_path="../configs", config_name="data_processing_config")
def process_data(config: DataProcessingConfig) -> None:
    logger = get_logger(Path(__file__).name)
    logger.info("Porcessing raw data...")

    processed_data_save_dir = config.processed_data_save_dir 

    cluster = instantiate(config.dask_cluster)
    client = Client(cluster)

    try:
    
        github_access_token = access_secret_version(config.infrastructure.project_id, config.github_acces_token_secret_id)

        get_raw_data_with_version(
            version=config.version,
            data_local_save_dir=config.data_local_save_dir,
            dvc_remote_repo=config.dvc_remote_repo,
            dvc_data_folder=config.dvc_data_folder,
            github_user_name=config.github_user_name,
            github_access_token=github_access_token,
        )

        dataset_reader_manager = instantiate(config.dataset_reader_manager)
        dataset_cleaner_manager = instantiate(config.dataset_cleaner_manager)
        
        df = dataset_reader_manager.read_data(config.dask_cluster.n_workers)
        
        print(60*'#')
        print(f"df.npartitions=")
        print(60*'#')

        logger.info("Cleaning data...")
        df = df.assign(cleaned_text=df.map_partitions(process_raw_data, dataset_cleaner_manager=dataset_cleaner_manager, meta=("text", "object")))
        df.compute()

        train_parquet_path = os.path.join(processed_data_save_dir, "train.parquet")
        dev_parquet_path = os.path.join(processed_data_save_dir, "dev.parquet")
        test_parquet_path = os.path.join(processed_data_save_dir, "test.parquet")

        df[df["split"] == "train"].to_parquet(train_parquet_path)
        df[df["split"] == "dev"].to_parquet(dev_parquet_path)
        df[df["split"] == "test"].to_parquet(test_parquet_path)        

        logger.info("Data processing finished!")

    finally:
        logger.info("Closing dask client and cluster...")
        client.close()
        cluster.close()

    # print(df.tail())
    # 'jtc' satırlarından birkaç örnek
    """
    # ---- DATAASETTEN DENEMELER ICIN ----
    twitter_df = df[df["dataset_name"].astype(str).str.strip().eq("twitter")][
        ["label", "dataset_name", "split", "text"]
    ]
    rows = twitter_df.head(5, npartitions=twitter_df.npartitions, compute=True)
    print("Twitter örnek satırlar:")
    print(rows.to_string(index=False) if not rows.empty else "— boş —")
    """

    """
    # Uniquue calculator
    names_df = df.map_partitions(
    lambda pdf: pdf[["dataset_name"]].drop_duplicates(),
    meta=pd.DataFrame({"dataset_name": pd.Series([], dtype="object")}),
    )

    unique_dataset_names = (
        names_df.drop_duplicates()
                .compute()            # burada artık concat simplify’a takılmaz
                ["dataset_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
    )
    print("DATASETS:", sorted(unique_dataset_names))
    """


if __name__ == "__main__":
    process_data()
