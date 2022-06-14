import json
import logging
import random
import math
from typing import Tuple

import click

from learning.rule_learner import *

logging.getLogger(__name__)


def split(examples: Dataset, split=0.8) -> Tuple[Examples, Examples]:
    indices = set(random.sample(range(len(examples)), math.ceil(split * len(examples))))
    train, eval = [], []
    for i, e in enumerate(examples):
        if i in indices:
            train.append(e)
        else:
            eval.append(e)
    return train, eval


@click.command()
@click.option('algos', '--algo', '-a', type=str, multiple=True,
              help='Names of the algorithms to test as registered by @AlgorithmRegistry.register')
@click.option('--dataset', '-d', type=str,
              help='path to the .json file containing the dataset. When loaded, this should be a dict two keys: '
                   '`pos` and `neg`.'
                   'the values should be positive and negative examples of the relation to learned with keys of '
                   'the examples as the variables of the target relation. For more examples look at the test cases'
                   'and their databases, e.g. tests/public/graph_small.json.')
@click.option('--kb', '-k', type=str,
              help='path to the .pl prolog knowledge base. Should be a knowledge base that you can load in prolog.')
@click.option('--eval', '-e', type=float, default=False,
              help='ratio used to split the positive examples and negative examples in training and evaluation data. '
                   'If omitted, no splitting is done and no evaluation is performed. If supplied, positive and '
                   'negative examples will be split according to the ratio, the rule learner algorithm'
                   ' will be trained on the training examples and the accuracy on the '
                   'positive and negative evaluation examples will be measured. Accuracy here means how many'
                   'of the positive examples the learned hypothesis will cover and how many of the negative '
                   'examples it will not cover')
@click.option('--target', '-t', type=str,
              help='Signature of the target predicate to learn in prolog notation. e.g.\'parent(X,Y).\'')
@click.option('--recursive', '-r', type=bool, is_flag=True,
              help='Whether the algorithm is supposed to learn recursive rules or not. Will be supplied as a keyword'
                   '`recursive` to the constructor of the trained/evaluated algorithm.')
@click.option('--random_seed', '-s', type=int, default=42,
              help='Random seed used for reproducibility. Best not to touch.')
def main(algos: List[str], dataset, kb, eval, target, recursive, random_seed):
    random.seed(random_seed)
    for algo_name in algos:
        with open(dataset, 'r') as f:
            data = json.load(f)
        if eval:
            # split examples
            train_pos, eval_pos = split(data['pos'], eval)
            train_neg, eval_neg = split(data['neg'], eval)
        else:
            train_pos = data['pos']
            train_neg = data['neg']
            eval_pos = []
            eval_neg = []

        ds = Dataset(positive_examples=train_pos, negative_examples=train_neg, kb=kb, target=target)
        with AlgorithmRegistry.create_algorithm(algo_name, dataset=ds, recursive=recursive) as algo:
            algo.fit()
            click.echo(f"Algorithm {click.style(algo_name, fg='green')} learned")
            click.echo(algo.hypothesis)
        if eval:
            acc_pos = sum(algo.predict(e) for e in eval_pos)
            acc_neg = sum(not algo.predict(e) for e in eval_neg)
            click.echo(f"Accuracy on positive examples: {acc_pos}/{len(eval_pos)} ({acc_pos / len(eval_pos):.2f})")
            click.echo(f"Accuracy on negative examples: {acc_neg}/{len(eval_neg)} ({acc_pos / len(eval_pos):.2f})")


if __name__ == '__main__':
    main()
