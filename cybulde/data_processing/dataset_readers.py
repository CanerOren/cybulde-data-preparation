from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import Optional
from cybulde.utils.utils import get_logger

import dask.dataframe as dd
from dask_ml.model_selection import train_test_split


class DatasetReader(ABC):
    required_columns = {"text", "label", "split", "dataset_name"}
    split_names = {"train", "dev", "test"}

    def __init__(self, dataset_dir: str, dataset_name: str) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name 


    def read_data(self) -> dd.DataFrame:
        train_df, dev_df, test_df = self._read_data()
        df = self.assign_split_names_to_data_frames_and_merge(train_df, dev_df, test_df)
        df["dataset_name"] = self.dataset_name
        if any(required_column not in df.columns.values for required_column in self.required_columns):
            raise ValueError(f"Dataset must contain all required columns: {self.required_columns}")
        unique_split_names = set(df["split"].unique().compute().tolist())
        if unique_split_names != self.split_names:
            raise ValueError(f"Dataset must contain all required split names: {self.split_names}")
        return df[list(self.required_columns)]


    @abstractmethod
    def _read_data(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        """
        Read and split dataset into 3 splits: train, dev, test.
        The return value must be a dd.DataFrame, with required columns: self.required_columns
        """


    def assign_split_names_to_data_frames_and_merge(self, train_df: dd.DataFrame, dev_df: dd.DataFrame, test_df: dd.DataFrame) -> dd.DataFrame:
        train_df["split"] = "train"
        dev_df["split"] = "dev"
        test_df["split"] = "test"
        return dd.concat([train_df, dev_df, test_df])
    

    def split_dataset(self, df: dd.DataFrame, test_size: float, stratify_column: Optional[str] = None) -> tuple[dd.DataFrame, dd.DataFrame]:
        if stratify_column is None:
            return train_test_split(df, test_size=test_size, random_state=1234, shuffle=True)

        # Stratify değerlerini somutla
        unique_vals = df[stratify_column].dropna().unique().compute().tolist()
        if not unique_vals:
            # Stratify edecek değer yoksa normal split
            return train_test_split(df, test_size=test_size, random_state=1234, shuffle=True)

        first_dfs = []
        second_dfs = []
        for val in unique_vals:
            sub_df = df[df[stratify_column] == val]
            sub_first_df, sub_second_df = train_test_split(
                sub_df, test_size=test_size, random_state=1234, shuffle=True
        )
        first_dfs.append(sub_first_df)
        second_dfs.append(sub_second_df)

    # Güvenli birleştirme
        if not first_dfs or not second_dfs:
            return train_test_split(df, test_size=test_size, random_state=1234, shuffle=True)

        first_df = dd.concat(first_dfs, interleave_partitions=True, ignore_unknown_divisions=True)
        second_df = dd.concat(second_dfs, interleave_partitions=True, ignore_unknown_divisions=True)
        return first_df, second_df


class GHCDatasetReader(DatasetReader):
    def __init__(self, dataset_dir: str, dataset_name: str, dev_split_ratio: float) -> None:
        super().__init__(dataset_dir, dataset_name) 
        self.dev_split_ratio = dev_split_ratio



    def _read_data(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        self.logger.info("Reading GHC dataset...")

        train_tsv_path = os.path.join(self.dataset_dir, "ghc_train.tsv")
        train_df = dd.read_csv(train_tsv_path, sep="\t", header=0)

        test_tsv_path = os.path.join(self.dataset_dir, "ghc_test.tsv")
        test_df = dd.read_csv(test_tsv_path, sep="\t", header=0)

        # label: hd+cv+vo > 0
        train_df["label"] = (
            (train_df["hd"] + train_df["cv"] + train_df["vo"]) > 0
        ).astype(int)
        test_df["label"] = (
            (test_df["hd"] + test_df["cv"] + test_df["vo"]) > 0
        ).astype(int)

        # train -> (train, dev) stratified by label
        train_df, dev_df = self.split_dataset(
            train_df,
            self.dev_split_ratio,
            stratify_column="label",
        )

        # ÜÇLÜ döndür: (train, dev, test)
        return train_df, dev_df, test_df

    
class JigsawToxicCommentsDatasetReader(DatasetReader):
    def __init__(self, dataset_dir: str, dataset_name: str, dev_split_ratio: float) -> None:
        super().__init__(dataset_dir, dataset_name) 
        self.dev_split_ratio = dev_split_ratio
        self.columns_for_label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def _read_data(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        self.logger.info(f"Reading {self.__class__.__name__}")
        
        test_csv_path = os.path.join(self.dataset_dir, "test.csv")
        test_df = dd.read_csv(test_csv_path)
        
        test_labels_csv_path = os.path.join(self.dataset_dir, "test_labels.csv")
        test_labels_df = dd.read_csv(test_labels_csv_path)

        test_df = test_df.merge(test_labels_df, on=["id"])
        test_df = test_df[test_df["toxic"] != -1]

        test_df = self.get_text_and_label_columns(test_df)

        train_csv_path = os.path.join(self.dataset_dir, "train.csv")
        train_df = dd.read_csv(train_csv_path)
        train_df = self.get_text_and_label_columns(train_df)

        train_df, dev_df = self.split_dataset(train_df, self.dev_split_ratio, stratify_column="label")

        return train_df, dev_df, test_df

    def get_text_and_label_columns(self, df: dd.DataFrame) -> dd.DataFrame:
        df["label"] = (df[self.columns_for_label].sum(axis=1) > 0).astype(int)
        df = df.rename(columns={"comment_text":"text"})
        return df
    

class DatasetReaderManager:
    def __init__(self, dataset_readers: dict[str, DatasetReader]) -> None:
        self.dataset_readers = dataset_readers

    def read_data(self) -> dd.DataFrame:
        dfs = [dataset_reader.read_data() for dataset_reader in self.dataset_readers.values()]
        df = dd.concat(dfs)
        return df