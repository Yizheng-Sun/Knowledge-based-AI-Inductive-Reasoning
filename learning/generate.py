import math
import random
from itertools import product
from typing import Dict, List, Any

from logging import getLogger

logger = getLogger(__name__)

from learning.util import Dataset
from learning.attr_learner import Disjunction


class AttributeDatasetGenerator:
    def __init__(self, attr_values: Dict[str, List[Any]], hypothesis: Disjunction):
        self.attr_values = attr_values
        self.hypothesis = hypothesis

    def generate_example(self) -> Dict[str, Any]:
        example = {k: random.choice(v) for k, v in self.attr_values.items()}
        example['target'] = self.hypothesis(example)

        return example

    def generate_all(self) -> List[Dict[str, Any]]:
        examples = [dict(e) for e in product(*[[(k, a) for a in v] for k, v in self.attr_values.items()])]
        for e in examples:
            e['target'] = self.hypothesis(e)
        return examples

    def __call__(self, num_examples: int = -1, balanced=False) -> Dataset:
        all_examples = self.generate_all()
        if balanced:
            pos = []
            neg = []
            for e in all_examples:
                if e['target']:
                    pos.append(e)
                else:
                    neg.append(e)
            logger.info(f'{len(pos)} positive examples.')
            logger.info(f'{len(neg)} negative examples.')
            max_examples = min(len(pos), len(neg))

            examples = random.sample(pos, min(math.floor(num_examples / 2), max_examples)) + \
                       random.sample(neg, min(math.ceil(num_examples / 2), max_examples))
            random.shuffle(examples)
        elif num_examples > 0:
            examples = random.sample(all_examples, k=num_examples)
        else:
            examples = all_examples
        return Dataset(examples, attr_values=self.attr_values)
