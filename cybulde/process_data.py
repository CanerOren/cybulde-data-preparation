# import pandas as pd
import os

from pathlib import Path

import dask.dataframe as dd

from dask.distributed import Client, wait
from hydra.utils import instantiate

from cybulde.configs_schemas.data_processing_config_schema import DataProcessingConfig
from cybulde.data_processing.dataset_cleaners import DatasetCleanerManager
from cybulde.utils.config_utils import custom_instantiate, get_pickle_config
from cybulde.utils.data_utils import filter_based_on_minimum_number_of_words
from cybulde.utils.io_utils import write_yaml_file
from cybulde.utils.utils import get_logger


def process_raw_data(df_partition: dd.DataFrame, dataset_cleaner_manager: DatasetCleanerManager) -> dd.Series:
    return df_partition["text"].apply(dataset_cleaner_manager)


@get_pickle_config(config_path="cybulde/configs/automatically_generated", config_name="data_processing_config")
def process_data(config: DataProcessingConfig) -> None:
    logger = get_logger(Path(__file__).name)
    logger.info("Porcessing raw data...")

    processed_data_save_dir = config.processed_data_save_dir

    cluster = custom_instantiate(config.dask_cluster)
    client = Client(cluster)

    try:
        dataset_reader_manager = instantiate(config.dataset_reader_manager)
        dataset_cleaner_manager = instantiate(config.dataset_cleaner_manager)

        df = dataset_reader_manager.read_data(config.dask_cluster.n_workers)

        logger.info("Cleaning data...")
        df = df.assign(
            cleaned_text=df.map_partitions(
                process_raw_data, dataset_cleaner_manager=dataset_cleaner_manager, meta=("text", "object")
            )
        )
        df = df.persist()
        wait(df)

        train_parquet_path = os.path.join(processed_data_save_dir, "train.parquet")
        dev_parquet_path = os.path.join(processed_data_save_dir, "dev.parquet")
        test_parquet_path = os.path.join(processed_data_save_dir, "test.parquet")

        train_df = df[df["split"] == "train"]
        dev_df = df[df["split"] == "dev"]
        test_df = df[df["split"] == "test"]

        train_df = filter_based_on_minimum_number_of_words(train_df, min_nrof_words=config.min_nrof_words)
        dev_df = filter_based_on_minimum_number_of_words(dev_df, min_nrof_words=config.min_nrof_words)
        test_df = filter_based_on_minimum_number_of_words(test_df, min_nrof_words=config.min_nrof_words)

        train_df.to_parquet(train_parquet_path)
        dev_df.to_parquet(dev_parquet_path)
        test_df.to_parquet(test_parquet_path)

        docker_info = {"docker_image": config.docker_image_name, "docker_tag": config.docker_image_tag}
        docker_info_save_path = os.path.join(processed_data_save_dir, "docker_info.yaml")

        write_yaml_file(docker_info_save_path, docker_info)

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
