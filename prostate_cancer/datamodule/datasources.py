# Copyright (c) The RationAI team.

from abc import ABC, abstractmethod
from typing import Self

import pandas


class BaseDataSource(ABC):
    """Abstract class for DataSource.

    It defines required methods and contains some fields that are common to all data
    sources.
    """

    splits: dict[str, float] | None
    stratified_keys: list[str] | None
    seed: int

    def __init__(
        self,
        seed: int,
        splits: dict[str, float] | None,
        stratified_keys: list[str] | None,
    ) -> None:
        self.seed = seed
        self.splits = self.__validate_splits(splits)
        self.stratified_keys = self.__validate_stratified_keys(stratified_keys)

    @staticmethod
    def __validate_splits(splits: dict[str, float] | None) -> dict[str, float] | None:
        if splits is None:
            return None

        if not (isinstance(splits, dict) or len(splits) > 0):
            raise ValueError(
                "splits must be a dictionary of split names and fractions or None"
            )

        return splits

    @staticmethod
    def __validate_stratified_keys(
        stratified_keys: list[str] | None,
    ) -> list[str] | None:
        if stratified_keys is None:
            return None
        elif isinstance(stratified_keys, list) and len(stratified_keys) > 0:
            return stratified_keys
        else:
            raise ValueError(
                "stratified_keys must be a list of column identifiers or None"
            )

    @abstractmethod
    def get_table(self) -> pandas.DataFrame:
        """Retrieves full dataset defined by this data source.

        Returns:
            pandas.DataFrame: Full dataset.
        """

    @abstractmethod
    def get_metadata(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Retrieves metadata for a dataset of entries defined by `data`.

        It is expected the `data` is a subset of the full dataset defined by this data
        source. The metadata values are retrieved based on a foreign key relationship
        between `data` and the metadata table.

        Args:
            data (pandas.DataFrame): Data to retrieve metadata for.

        Returns:
            pandas.DataFrame: New dataframe with metadata columns.
        """

    @abstractmethod
    def split(self) -> dict[str, Self] | Self:
        """Splits datasource into N parts, where N is `len(splits)`.

        The size of each split is defined by the values in the `splits` dictionary.
        The keys are used to name the splits.

        Returns:
            dict[str, BaseDataSource] | BaseDataSource: split data sources.
        """
