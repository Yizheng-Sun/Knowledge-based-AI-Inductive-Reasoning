import random
import timeit
import math
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy.stats import fisher_exact, t
import click

from learning.attr_learner import *

from learning.attr_learner import Disjunction, Conjunction
from learning.generate import AttributeDatasetGenerator
from learning.util import Algorithm, Dataset, AlgorithmRegistry, Examples


def evaluate(algo: Algorithm, eval: Dataset) -> float:
    """

    Returns: how many were answered correctly.

    """
    preds = algo.predict_all(eval)
    acc = sum(p == e['target'] for p, e in zip(preds, eval))
    return acc


def generate_random_hypothesis(attr_values: Dict[str, List[bool]], max_relevant=None, max_conj=4):
    """
    Generates a random hypothesis.

    Args:
        attr_values: attributes and their possible values
        max_relevant: how many of those should be part of hypothesis
        max_conj: how long the overall disjunction should be

    Returns:

    """
    relevant_attrs = random.sample(list(attr_values.items()), max_relevant)
    conjs = []
    for _ in range(max_conj):
        num_attr_for_this_branch = len(relevant_attrs)
        attrs_for_this_branch = random.sample(relevant_attrs, num_attr_for_this_branch)
        conj = Conjunction({k: random.choice(v) for k, v in attrs_for_this_branch})
        while conj in conjs:
            conj = Conjunction({k: random.choice(v) for k, v in attrs_for_this_branch})
        conjs.append(conj)
    return Disjunction(conjs)


@click.command(help='How well the attribute based learner learns from noisy data.'
                    'First generates a dataset from a randomly generated hypothesis of a given size, then randomly'
                    'corrupts a given number of examples by swapping the labels, then trains given algorithms on '
                    'the corrupted training set. Then swaps the labels back and evaluates how many examples '
                    'the algorithms can predict correctly. Compares whether the differences results between algorithms '
                    'are statistically significant.')
@click.option('algos', '--algo', '-a', type=str, multiple=True,
              help='Names of the algorithms to test as registered by @AlgorithmRegistry.register. '
                   'Append multiple options'
                   ' (e.g. [...] -a algo1 -a algo2 -a algo3 [...]) to evaluate multiple algorithms. '
                   'To compare the perfromance, all subsequent algorithms will be compared to the first algo '
                   '(e.g. algo2 to algo1 and algo3 to algo1).')
@click.option('sizes', '--size', '-s', type=int, multiple=True, default=[5, 7, 10],
              help='Sizes of datasets as determined by the number of attributes. Will generate all possible '
                   'configurations of attributes. Attributes are binary, so the size of the dataset will be 2**n for'
                   'n attributes. Append multiple sizes (e.g. -s 5 -s 7) to test algorithms for multiple dataset'
                   ' sizes (e.g. 2**5=32 and 2**7=128)')
@click.option('max_dets', '--max-det', '-d', type=int, multiple=True, default=[3],
              help='How many of the attributes will the (randomly generated) true function will depend on '
                   '(consistent determinations). '
                   'Append multiple numbers (e.g. -d 3 -d 10) to test algorithms for multiple consistent determination '
                   ' sizes (e.g. true function will depend on values of 3 and 10 attributes respectively.)')
@click.option('corrupts', '--corrupt_ratio', '-c', type=float, multiple=True, default=[0.1, 0.3, 0.5],
              help='"Corrupts" this many training examples by randomly swapping the label. Before evaluating, swaps'
                   'the label back. '
                   'Append multiple numbers (e.g. -c 0.3 -c 0.5) to test algorithms for multiple ratios. '
                   ' sizes (e.g. 30% and 50% of training examples will be randomly swapped.')
@click.option('--random_seed', '-r', type=int, default=42,
              help='Random seed used for reproducibility. Best not to touch.')
def eval_noisy(algos: List[str], max_dets: List[int], corrupts: List[float], sizes: List[int], random_seed=42):
    random.seed(random_seed)
    for size in sizes:
        for corrupt in corrupts:
            for max_det in max_dets:

                attr_values = {f"A{i}": [True, False] for i in range(size)}
                hypothesis = generate_random_hypothesis(attr_values, max_det, 2 ** (max_det - 1))
                generator = AttributeDatasetGenerator(attr_values=attr_values, hypothesis=hypothesis)
                dataset = Dataset(generator.generate_all())
                # corrupt
                corrupt_indices = set(
                    random.sample(range(len(dataset.examples)), math.ceil(corrupt * len(dataset))))
                click.echo(20 * "====")
                click.echo(
                    f"Number of attributes: {size}. of which {max_det} are used to generate a random hypothesis. "
                    f"Dataset size: {2 ** size}, number of corrupted examples {len(corrupt_indices)} ")

                for i, e in enumerate(dataset.examples):
                    if i in corrupt_indices:
                        e['target'] = not e['target']

                algorithms = [AlgorithmRegistry.create_algorithm(a, dataset=dataset) for a in algos]
                for algo in algorithms:
                    algo.fit()
                # un-corrupt
                for i, e in enumerate(dataset.examples):
                    if i in corrupt_indices:
                        e['target'] = not e['target']
                results = []
                for algo in algorithms:
                    result = evaluate(algo, dataset)
                    results.append(result)
                if results:
                    baseline, *improvements = list(zip(algos, results))
                    b_algo, b_score = baseline
                    b_acc = b_score / 2 ** size
                    click.echo(20 * "----")
                    click.echo(
                        f"{click.style(b_algo, fg='green')}(baseline) accuracy: {b_score}/{2 ** size} ({b_acc:.3f}))")
                    for algo, score in improvements:
                        acc = score / 2 ** size
                        odds, p_value = fisher_exact([[b_score, 2 ** size - b_score], [score, 2 ** size - score]])
                        click.echo(f"Accuracy of {click.style(algo, fg='green')} \t {score}/{2 ** size} ({acc:.3f})) | "
                                   f"Statistical significance at p=0.05? {click.style(str(p_value < 0.05), bold=True)}")


def split(dataset: Dataset, split=0.8) -> Tuple[Examples, Examples]:
    indices = set(random.sample(range(len(dataset.examples)), math.ceil(split * len(dataset))))
    train, eval = [], []
    for i, e in enumerate(dataset):
        if i in indices:
            train.append(e)
        else:
            eval.append(e)
    return train, eval


@click.command(help='How well the attribute based learner learns from less examples.'
                    'First generates a dataset from a randomly generated hypothesis of a given size, then randomly'
                    'splits the dataset in training and evaluation data based on a given split ratio then '
                    'trains the given algorithms on the training set. and evaluates how many left out examples '
                    'the algorithms can predict correctly. Compares whether the differences results between algorithms '
                    'are statistically significant.')
@click.option('algos', '--algo', '-a', type=str, multiple=True,
              help='Names of the algorithms to test as registered by @AlgorithmRegistry.register. '
                   'Append multiple options'
                   ' (e.g. [...] -a algo1 -a algo2 -a algo3 [...]) to evaluate multiple algorithms. '
                   'To compare the perfromance, all subsequent algorithms will be compared to the first algo '
                   '(e.g. algo2 to algo1 and algo3 to algo1).')
@click.option('sizes', '--size', '-s', type=int, multiple=True, default=[5, 7, 10],
              help='Sizes of datasets as determined by the number of attributes. Will generate all possible '
                   'configurations of attributes. Attributes are binary, so the size of the dataset will be 2**n for'
                   'n attributes. Append multiple sizes (e.g. -s 5 -s 7) to test algorithms for multiple dataset'
                   ' sizes (e.g. 2**5=32 and 2**7=128)')
@click.option('max_dets', '--max-det', '-d', type=int, multiple=True, default=[10],
              help='How many of the attributes will the (randomly generated) true function will depend on '
                   '(consistent determinations). '
                   'Append multiple numbers (e.g. -d 3 -d 10) to test algorithms for multiple consistent determination '
                   ' sizes (e.g. true function will depend on values of 3 and 10 attributes respectively.)')
@click.option('train_splits', '--train_split', '-t', type=float, multiple=True, default=[0.1, 0.3, 0.5])
@click.option('--random_seed', '-r', type=int, default=42,
              help='Random seed used for reproducibility. Best not to touch.')
def eval_size(algos: List[str], max_dets: List[int], train_splits: List[float], sizes: List[int], random_seed=42):
    random.seed(random_seed)

    for size in sizes:
        for split_ratio in train_splits:
            for max_det in max_dets:
                max_det = min(max_det, size)
                results = []
                attr_values = {f"A{i}": [True, False] for i in range(size)}
                hypothesis = generate_random_hypothesis(attr_values, max_det, 2 ** (max_det - 1))
                generator = AttributeDatasetGenerator(attr_values=attr_values, hypothesis=hypothesis)
                dataset = Dataset(generator.generate_all())

                train, eval = split(dataset, split_ratio)
                # corrupt
                click.echo(20 * "====")
                click.echo(
                    f"Number of attributes: {size}. of which {max_det} are used to generate a random hypothesis. "
                    f"Dataset size: {2 ** size}, size of training set {len(train)}. ")
                click.echo(f"{sum(e['target'] == True for e in dataset.examples)} of which are True.")
                algorithms = [AlgorithmRegistry.create_algorithm(a, dataset=Dataset(train)) for a in algos]
                for algo in algorithms:
                    algo.fit()

                for algo in algorithms:
                    result = evaluate(algo, Dataset(eval))
                    results.append(result)
                if results:
                    baseline, *improvements = list(zip(algos, results))
                    b_algo, b_score = baseline

                    b_acc = b_score / len(eval)
                    click.echo(20 * "----")
                    click.echo(
                        f"{click.style(b_algo, fg='green')}(baseline) accuracy: {b_score}/{len(eval)} ({b_acc:.3f}))")
                    for algo, score in improvements:
                        acc = score / len(eval)
                        odds, p_value = fisher_exact([[b_score, len(eval) - b_score], [score, len(eval) - score]])
                        click.echo(f"Accuracy of {click.style(algo, fg='green')} \t {score}/{len(eval)} ({acc:.3f})) | "
                                   f"Statistically significant at p=0.05? {click.style(str(p_value < 0.05), bold=True)}")


def get_mean_var_ci(sample, alpha=0.025):
    sample = np.array(sample)
    t_ci = t.ppf(1 - alpha, df=len(sample) - 1)
    return sample.mean(), sample.var(), t_ci * sample.std() / math.sqrt(len(sample))


@click.command(help='This evaluates the runtime of the attribute based learner. '
                    'First generates a dataset from a randomly generated hypothesis of a given size, then trains the '
                    'given algorithms for a number of times. Records the average training time and the confidence interval '
                    'at 95%.')
@click.option('algos', '--algo', '-a', type=str, multiple=True,
              help='Names of the algorithms to test as registered by @AlgorithmRegistry.register. '
                   'Append multiple options'
                   ' (e.g. [...] -a algo1 -a algo2 -a algo3 [...]) to evaluate multiple algorithms. '
                   'To compare the performance, all subsequent algorithms will be compared to the first algo '
                   '(e.g. algo2 to algo1 and algo3 to algo1).')
@click.option('sizes', '--size', '-s', type=int, multiple=True, default=[5, 7, 10],
              help='Sizes of datasets as determined by the number of attributes. Will generate all possible '
                   'configurations of attributes. Attributes are binary, so the size of the dataset will be 2**n for'
                   'n attributes. Append multiple sizes (e.g. -s 5 -s 7) to test algorithms for multiple dataset'
                   ' sizes (e.g. 2**5=32 and 2**7=128)')
def eval_time(algos: List[str], sizes: List[int]):
    for size in sizes:
        num_runs = int((2 ** 15) / (size ** 3))
        # number = int((2 ** 20) / ((size) ** 3))
        click.echo(20 * "====")
        click.echo(f"Number of attributes: {size}. all of which are used to generate a random hypothesis. "
                   f"Dataset size: {2 ** size}. running {num_runs} iterations.")
        click.echo(20 * "----")
        for algo_name in algos:
            results = []
            for _ in range(num_runs):
                attr_values = {f"A{i}": [True, False] for i in range(size)}
                hypothesis = generate_random_hypothesis(attr_values, size, 2 ** (size - 1))
                generator = AttributeDatasetGenerator(attr_values=attr_values, hypothesis=hypothesis)
                dataset = Dataset(generator.generate_all())
                algo = AlgorithmRegistry.create_algorithm(algo_name, dataset=dataset)
                algo.fit()

                results.append(timeit.timeit("algo.fit()", globals={'algo': algo}, number=5) / 5)

            mean, var, ci = get_mean_var_ci(results)
            click.echo(f"{click.style(algo_name, fg='green')} \t {mean * 1000:.5f}ms +/- {ci * 1000:.3f} ms")


if __name__ == '__main__':
    cli = click.Group(help='This is the evaluation script for evaluating attribute-based learners. The three metrics '
                           'are accuracy when facing noisy data, accuracy when facing little training data and '
                           'training time. The three subcommands evaluate these three metrics.'
                           'run python evaluate_attributes.py [subcommand] --help to get more info.')
    cli.add_command(eval_size)
    cli.add_command(eval_noisy)
    cli.add_command(eval_time)
    cli()
