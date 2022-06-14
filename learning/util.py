from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Dict, Callable, Any, List, Sequence

from logging import getLogger
from tabulate import tabulate

logger = getLogger(__name__)

Example = Dict[str, Any]
Examples = List[Example]


class Dataset(Sequence):

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)

    examples: Examples
    attributes: List[str]
    attr_values: Dict[str, List[Any]]

    def __init__(self, examples: Examples = None, positive_examples=None, negative_examples=None, attr_values=None,
                 kb=None, target=None):
        assert examples or (positive_examples and negative_examples), \
            "Need examples! Either as whole or as positive+negative"
        assert not (examples and (positive_examples or negative_examples)), \
            "Can't have both fullexamples and positive+negative split"
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.examples = examples or positive_examples + negative_examples
        self._attr_values = {k: v for k, v in (attr_values or {}).items() if k != 'target'} or None
        self.attributes = list(self.attr_values.keys())
        self.kb = kb
        self.target = target

    @classmethod
    def from_csv(cls, path: str) -> 'Dataset':
        ...

    def to_csv(self, path: str):
        ...

    @property
    def attr_values(self):
        if self._attr_values is None:
            self._attr_values = defaultdict(set)
            for e in self.examples:
                for k, v in e.items():
                    self._attr_values[k].add(v)
                if 'target' in self._attr_values:
                    self._attr_values.pop('target')
        return self._attr_values


class Algorithm(ABC):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def find_hypothesis(self):
        ...

    def fit(self):
        self.hypothesis = self.find_hypothesis()

    def predict(self, example: Example) -> bool:
        return self.hypothesis(example)

    def predict_all(self, dataset: Dataset) -> List[bool]:
        return [self.predict(e) for e in dataset.examples]


class AlgorithmRegistry:
    """ The factory class for creating executors"""

    registry: Dict[str, Algorithm] = {}
    """ Internal registry for available Algorithms """

    @classmethod
    def get_name(cls, algo: Algorithm) -> str:
        for k, v in cls.registry.items():
            if v == algo.__class__:
                return k
        raise ValueError(f"{algo.__class__.__name__} is not registered!")

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Algorithm) -> Algorithm:
            if name in cls.registry:
                logger.warning('Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_algorithm(cls, name: str, **kwargs) -> Algorithm:
        """ Factory command to create the executor """

        exec_class = cls.registry.get(name, False)
        if not exec_class:
            raise ValueError(f"No implementation with name {name} exists. "
                             f"Did you forget to @AlgorithmRegistry.register it?")
        else:
            executor = exec_class(**kwargs)
            return executor


def tabulate_dataset(examples, target_name):
    headings = examples[0].keys()
    rows = []
    for example in examples:
        rows.append([example[h] for h in headings])
    # headings
    print(tabulate(rows, headers=[h.replace('target', target_name) for h in headings]))
