# import pandas as pd
from hydra.utils import instantiate

from cybulde.configs_schemas.data_processing_config_schema import DataProcessingConfig
from cybulde.utils.config_utils import get_config
from cybulde.utils.data_utils import get_raw_data_with_version
from cybulde.utils.gcp_utils import access_secret_version


@get_config(config_path="../configs", config_name="data_processing_config")
def process_data(config: DataProcessingConfig) -> None:
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
    df = dataset_reader_manager.read_data().compute()
    sample_df = df.sample(n=5)

    for _, row in sample_df.iterrows():
        text = row["text"]
        cleaned_text = dataset_cleaner_manager(text)

        print(60 * "#")
        print(f"{text=}")
        print(60 * "#")
        print(f"{cleaned_text=}")

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
