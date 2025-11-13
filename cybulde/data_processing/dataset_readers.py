from __future__ import annotations

import os

from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

import dask.dataframe as dd
import pandas as pd

from dask_ml.model_selection import train_test_split

from cybulde.utils.utils import get_logger
from cybulde.utils.data_utils import repartition_dataframe

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

        # 1) Erken tetikleyici: expr optimize sırasında split kolonunu gerçekten “görsün”
        _ = df[["split"]].head(1, npartitions=df.npartitions, compute=True)

        # 2) Zorlamalı unique: map_partitions + drop_duplicates + union
        splits_df = cast(
            dd.DataFrame,
            df.map_partitions(  # type: ignore[no-untyped-call]
                lambda pdf: pdf[["split"]].drop_duplicates(),
                meta=pd.DataFrame({"split": pd.Series([], dtype="object")}),
            ),
        )
        unique_split_names = set(
            splits_df["split"].dropna().astype(str).str.strip().drop_duplicates().compute().tolist()
        )
        if any(c not in df.columns.values for c in self.required_columns):
            raise ValueError(f"Dataset must contain all required columns: {self.required_columns}")

        if unique_split_names != self.split_names:
            raise ValueError(f"Dataset must contain all required split names: {self.split_names}")

        return cast(dd.DataFrame, df[list(self.required_columns)])

    @abstractmethod
    def _read_data(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        """
        Read and split dataset into 3 splits: train, dev, test.
        The return value must be a dd.DataFrame, with required columns: self.required_columns
        """

    def assign_split_names_to_data_frames_and_merge(
        self, train_df: dd.DataFrame, dev_df: dd.DataFrame, test_df: dd.DataFrame
    ) -> dd.DataFrame:
        train_df = train_df.assign(split="train")
        dev_df = dev_df.assign(split="dev")
        test_df = test_df.assign(split="test")

        frames = [train_df, dev_df, test_df]

        # Union meta
        meta_cols: dict[str, pd.Series] = {}
        for f in frames:
            for col, dtype in f._meta.dtypes.items():
                meta_cols[col] = pd.Series([], dtype=dtype)
        meta_cols.setdefault("split", pd.Series([], dtype="object"))
        meta = pd.DataFrame(meta_cols)

        return cast(
            dd.DataFrame,
            dd.concat(  # type: ignore[no-untyped-call]
                frames,
                interleave_partitions=True,
                ignore_unknown_divisions=True,
                meta=meta,  # ← kritik
            ),
        )

    def split_dataset(
        self, df: dd.DataFrame, test_size: float, stratify_column: Optional[str] = None
    ) -> tuple[dd.DataFrame, dd.DataFrame]:
        if stratify_column is None:
            return cast(
                Tuple[dd.DataFrame, dd.DataFrame],
                train_test_split(df, test_size=test_size, random_state=1234, shuffle=True),
            )

        # Stratify değerlerini somutla
        unique_vals = (
            df[stratify_column]
            .map_partitions(lambda s: s.dropna().drop_duplicates(), meta=pd.Series([], dtype="object"))
            .drop_duplicates()
            .compute()
            .tolist()
        )

        if not unique_vals:
            # Stratify edecek değer yoksa normal split
            return cast(
                Tuple[dd.DataFrame, dd.DataFrame],
                train_test_split(df, test_size=test_size, random_state=1234, shuffle=True),
            )

        first_dfs = []
        second_dfs = []
        for val in unique_vals:
            sub_df = df[df[stratify_column] == val]
            sub_first_df, sub_second_df = train_test_split(sub_df, test_size=test_size, random_state=1234, shuffle=True)
            first_dfs.append(sub_first_df)
            second_dfs.append(sub_second_df)

        # Güvenli birleştirme
        if not first_dfs or not second_dfs:
            return cast(
                Tuple[dd.DataFrame, dd.DataFrame],
                train_test_split(df, test_size=test_size, random_state=1234, shuffle=True),
            )

        first_df = cast(
            dd.DataFrame,
            dd.concat(
                first_dfs, interleave_partitions=True, ignore_unknown_divisions=True  # type: ignore[no-untyped-call]
            ),
        )
        second_df = cast(
            dd.DataFrame,
            dd.concat(
                second_dfs, interleave_partitions=True, ignore_unknown_divisions=True  # type: ignore[no-untyped-call]
            ),
        )
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
        train_df["label"] = ((train_df["hd"] + train_df["cv"] + train_df["vo"]) > 0).astype(int)
        test_df["label"] = ((test_df["hd"] + test_df["cv"] + test_df["vo"]) > 0).astype(int)

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
        df = df.rename(columns={"comment_text": "text"})
        return df


class TwitterDatasetReader(DatasetReader):
    def __init__(self, dataset_dir: str, dataset_name: str, test_split_ratio: float, dev_split_ratio: float) -> None:
        super().__init__(dataset_dir, dataset_name)
        self.test_split_ratio = test_split_ratio
        self.dev_split_ratio = dev_split_ratio

    def _read_data(self) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        self.logger.info(self.__class__.__name__)
        csv_path = os.path.join(self.dataset_dir, "cyberbullying_tweets.csv")
        df = dd.read_csv(csv_path)

        df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})
        df["label"] = (df["label"] != "not_cyberbullying").astype(int)

        train_df, test_df = self.split_dataset(df, self.test_split_ratio, stratify_column="label")
        train_df, dev_df = self.split_dataset(train_df, self.dev_split_ratio, stratify_column="label")

        return train_df, dev_df, test_df


class DatasetReaderManager:
    def __init__(self, dataset_readers: dict[str, DatasetReader], repartition: bool = True) -> None:
        self.dataset_readers = dataset_readers
        self.repartition = repartition

    def read_data(self, nrof_workers: int) -> dd.DataFrame:
        dfs = [dr.read_data() for dr in self.dataset_readers.values()]

        # Union meta (ihtiyaten)
        meta_cols: dict[str, pd.Series] = {}
        for f in dfs:
            for col, dtype in f._meta.dtypes.items():
                meta_cols[col] = pd.Series([], dtype=dtype)
        for must in ("text", "label", "split", "dataset_name"):
            meta_cols.setdefault(must, pd.Series([], dtype="object"))
        meta = pd.DataFrame(meta_cols)
        df = cast(
            dd.DataFrame,
            dd.concat(  # type: ignore[no-untyped-call]
                dfs,
                interleave_partitions=True,
                ignore_unknown_divisions=True,
                meta=meta,
            ),
        )

        if self.repartition:
            df = repartition_dataframe(
                df,
                nrof_workers = nrof_workers
            )

        return df