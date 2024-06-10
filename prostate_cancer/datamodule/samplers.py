# Copyright (c) The RationAI team.

"""TreeSampler is a default sampling data structure.

The inner nodes of a TreeSampler contain column names. The number of outgoing edges
from an inner node is equal to the number of unique values appearing in that column.
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import Self

import numpy as np
import pandas

from prostate_cancer.datamodule.datasources import BaseDataSource


log = logging.getLogger("sampler")


class Node:
    """Node object for SamplingTree.

    Splitting a node on a column with G unique values partitions
    the DataFrame into G new DataFrames. For each new DataFrame a
    child Node is created. Split Node content is erased to free memory.

    All children nodes are chained via 'next' reference to allow
    quick traversal of all leaf nodes.
    """

    node_name: str
    data: pandas.DataFrame | None
    parent: Self | None = None
    children: list[Self]
    next: Self | None = None
    metadata: dict

    def __init__(self, name: str, df: pandas.DataFrame) -> None:
        self.node_name = name
        self.data = df
        self.children = []
        self.metadata = {}

    def split_node(self, column: str) -> None:
        """Leaf node is replaced with an internal node and N new leaf nodes, where N is the number of unique values for column `col`.

        The dataframe in the original leaf node is split in such a way that
        there is a single unique value in the split column in each of the new
        leaf node. The values across leaf nodes all are different.

        Args:
            column (str): Column name.
        """
        partitions = {col_val: df for col_val, df in self.data.groupby(column)}  # noqa: C416
        for column_val, df in partitions.items():
            new_node = Node(f"{self.node_name}/{column_val}", df)
            new_node.parent = self
            self.children.append(new_node)

        for node, next_node in zip(self.children[:-1], self.children[1:], strict=False):
            node.next = next_node

        self.data = None

    def __repr__(self) -> str:
        return f"Node({self.node_name})"


class SamplingTree:
    """DataStructure for sampling.

    - The root node holds reference to the left-most leaf.
    - The leaves of a SamplingTree are singly linked, allowing
    quick one-way traversal of all leaves.
    - Only leaf nodes of a SamplingTree holds data.
    - On column split, each leaf node introduces G new children nodes, where G
      is the number of unique values in the column. These new nodes form a new
      tree level.
    - The order of columns in which the SamplingTree defines the final form
      of the SamplingTree.
    """

    root: Node
    leftmost_leaf: Node
    split_columns: list[str]

    def __init__(self, df: pandas.DataFrame) -> None:
        root_node = Node("/ROOT", df)
        self.root = root_node
        self.leftmost_leaf = root_node
        self.split_columns = []

    def split(self, columns: str | list[str]) -> None:
        """Creates a new level of a SamplingTree.

        Each node receives number of new children equal to unique values in the
        selected column.

        Args:
            columns (str | list[str]): Column name or list of column names.

        Returns:
            None
        """
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            self._split_on_column(column)

    def _split_on_column(self, column: str) -> None:
        # Idempotent operation
        if column in self.split_columns:
            return

        # Check if column exists
        cur_node = self.leftmost_leaf
        if column not in cur_node.data:
            raise ValueError(f"Column {column} does not exist.")

        # Split all leaves and create one new level
        cur_node.split_node(column)
        while cur_node.next is not None:
            prev_node, cur_node = cur_node, cur_node.next
            cur_node.split_node(column)
            prev_node.children[-1].next = cur_node.children[0]
            prev_node.next = None
        self.leftmost_leaf = self.leftmost_leaf.children[0]
        self.split_columns.append(column)


class BaseSampler(ABC):
    """Base class for a sampler."""

    @abstractmethod
    def build_inner_structure(self, data_source: BaseDataSource) -> None: ...

    @abstractmethod
    def get_sample(self) -> list[dict]:
        """Defines sampling strategy for a Sampler.

        Actually samples data from the data source. It is intended to be called by a
        Generator on each epoch end (possibly) or when we need to resample the data.
        """


class TreeSampler(BaseSampler, ABC):
    """TreeSampler is a sampler that utilizes SamplingTree data structure.

    The SamplingTree is a tree data structure that is used to sample data
    from a dataset. The tree is built by splitting the dataset on a set of
    columns. The columns are split in the order they are provided in the
    configuration file. The leaf nodes of the tree contain the actual data
    samples.

    sampling_tree (SamplingTree): SamplingTree data structure.
    index_levels (list[str]): List of column names appearing in the input DataFrames.
                              These column names are then used to multi-level sampling
                              tree. Column names are processed in order of appearance.
    """

    sampling_tree: SamplingTree
    index_levels: list[str]

    def __init__(self, index_levels: list[str]) -> None:
        self.index_levels = index_levels

    def build_inner_structure(self, data_source: BaseDataSource) -> None:
        """Builds a SamplingTree from the provided BaseDataSource.

        Args:
            data_source (BaseDataSource): BaseDataSource containing paths to input files

        Returns:
            SamplingTree: SamplingTree data structure.
        """
        tiles = data_source.get_table()
        dataset = data_source.get_metadata(tiles)

        sampling_tree = SamplingTree(dataset)
        sampling_tree.split(self.index_levels)
        self.sampling_tree = sampling_tree


class RandomTreeSampler(TreeSampler):
    """RandomSampler samples randomly 'epoch_size' entries.

    Supports multi-level sampling by including 'index_level'.

    For each sampled entry, the algorithm starts at the root and randomly
    follows a path until it reaches a leaf where a row is randomly
    selected. This mechanism can help to deal with over/under-represented
    entries.

    Attributes:
        _rng (numpy.random.default_rng): Seeded random state.
    """

    epoch_size: int
    seed: int
    _rng: np.random.Generator

    def __init__(self, index_levels: list[str], seed: int, epoch_size: int) -> None:
        super().__init__(index_levels)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.epoch_size = epoch_size

    def get_sample(self) -> list[dict]:
        """Returns a list of sampled entries of size equal to the `configured epoch_size`.

        At every node a child branch is chosen uniformly from all children of
        the current node.

        Returns:
            pandas.DataFrame:
        """
        epoch_samples = self._sample_node(self.epoch_size, self.sampling_tree.root)
        epoch_samples_new = epoch_samples.sample(
            frac=1, random_state=self._rng.bit_generator
        ).reset_index(drop=True)
        return epoch_samples_new.to_dict("records")

    def _sample_node(self, num_entries: int, node: Node) -> pandas.DataFrame:
        """Specifies sampling algorithm.

        The algorithm simulates generating epoch_size paths through the
        sampling tree. Each leaf node is visited at most once.

        In a ROOT node the algorithm determines the number of traversals
        through each child. The sum of traversals must be equal to the
        num_entries=epoch_size.

        In each inner node, the algorithm repeats the process. However, the
        sum of traversals for inner-node's children must be equal to the
        number of traversals computed by the node's parent.

        In each leaf node rows are sampled randomly from the data table.

        Args:
            num_entries (int): number of paths leading through current node.
            node (Node): current node

        Returns:
            DataFrame:
        """
        if not node.children:
            return node.data.sample(
                n=num_entries,
                replace=True,
                random_state=self._rng.bit_generator,
            )

        sampled_nodes = self._rng.choice(
            a=node.children, size=num_entries, replace=True
        )
        cur_counts: Counter[Node] = Counter(sampled_nodes)

        result = []
        for child_node, child_num_entries in cur_counts.items():
            result.append(
                self._sample_node(num_entries=child_num_entries, node=child_node)
            )
        return pandas.concat(result)


class SequentialTreeSampler(TreeSampler):
    """SequentialSampler traverses all leaves once and returns their data content.

    Supports multi-level sampling by including 'index_level'.

    The algorithm starts at the left-most leaf and returns its entire
    content. The SequentialTreeSampler moves to the next leaf when the
    ``next()`` function is called.

    Attributes:
        active_node (Node): A pointer to current leaf node.
        advance_to_next (bool): Boolean value deciding whether to advance to the next
            node when generating samples.
    """

    active_node: Node
    advance_to_next: bool

    def __init__(self, index_levels: list[str], advance_to_next: bool) -> None:
        super().__init__(index_levels=index_levels)
        self.advance_to_next = advance_to_next

    def build_inner_structure(self, data_source: BaseDataSource) -> None:
        """Builds the sampling tree and sets the active node to the left-most leaf.

        Args:
            data_source (BaseDataSource): Data source.
        """
        super().build_inner_structure(data_source)
        self.active_node = self.sampling_tree.leftmost_leaf

    def get_sample(self) -> list[dict] | None:
        """Returns the content of currently active SamplerTree node.

        Returns:
            list[dict] | None: List of sampled entries.
        """
        res = None if self.active_node is None else self.active_node.data

        if self.advance_to_next:
            self.next()
            log.debug("Sampler advanced to next node...")

        res = res.to_dict("records")
        return res

    def next(self) -> None:
        """Sets next leaf as an active node."""
        if self.active_node is None:
            raise StopIteration
        else:
            self.active_node = self.active_node.next
