# Readme for evaluation scripts

The repo contains three scripts that should help you going with Exercise 3. You can modify them if you want to show
something that they don't cover, but make sure to document what you did so it's understandable to us. The tree scripts
are `evaluate_attributes.py`, `evaluate_rules.py` and
`generate_graphs.py`.

## Improving the algorithms

For Exercise 3, you will need to provide your own, improved implementations of one
(or multiple) algorithms. Please don't overwrite the base algorithms, implemented in Exercise 1 and 2, as they will be
used for comparison, that your algorithm is in fact better. If your changes improve upon an existing algorithm, the best
thing is to subclass it. For example, if you want to improve the `DecisionTreeLearner`
algorithm, you can do the following in the file `learning/attr_learner.py`:

```python
@AlgorithmRegistry.register("my-dtl")
class MyDecisionTreeLearner(DecisionTreeLearner):
    ...
```

and then override the methods that you want to improve, e.g.

```python
@AlgorithmRegistry.register("my-dtl")
class MyDecisionTreeLearner(DecisionTreeLearner):
    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        ...
```

If instead you want to implement a completely new algorithm, you will need to subclass `Algorithm`m i.e.:

```python
@AlgorithmRegistry.register("my-algo")
class MyAwesomeAlgorithm(Algorithm):
    def find_hypothesis(self):
        ...
```

in this case, you _need_ to implement method `find_hypothesis`.

For the foil example, it is similar, just subclass the `_FOIL` class. The only intricacy with foil is because we use it
with a context manager, you will need to do something similar to the following:

```python
@AlgorithmRegistry.register('my-foil')
@contextmanager
def MyFOIL(dataset: Dataset, recursive=False):
    f = _MyFOIL(dataset, recursive)
    try:
        yield f
    finally:
        f.abolish()  # this cleans up the kb after you're done using your algo


class _MyFOIL(_FOIL):
    prolog: Prolog
```

so you will need to implement the functionality in `_MyFOIL`, which will subclass
`_FOIL`, but in order to expose it to the scripts, you will need to create a function
`MyFOIL` which has to have the structure as above and be decorated with the
`@contextmanager` decorator which has to be then decorated with the `@AlgorithmRegistry.register` (in that order!).

In both cases, don't forget to `@AlgorithmRegistry.register('name')` your algorithm because that is what exposes it to
the evaluation scripts.

## Evaluate the attibute-based learner

`evaluate_attributes.py` has three subcommands. to learn about them, run

```bash
python evaluate_attributes.py --help
```

the tree commands are `eval-noisy`, `eval-size` and `eval-time`, which generate some pseudo-random data and hypotheses,
based on given parameters, train supplied algorithms and evaluate according to the corresponding metric.

### Evaluate on noisy data

run

```bash
python evaluate_attributes.py eval-noisy --help
```

to understand what all the options do. Once you implemented a version of the algorithm for which you think your
algorithm performs better than the baseline developed in exercise 1, you can test it by running, e.g.

```bash
python evaluate_attributes.py eval-noisy -a dtl -a my-dtl -c 0.2 -c 0.5 -s 10 -d 10
```

This will run two experiments for both of the algorithms: the script will generate a (pseudo-)random dataset of the size
2^10=1024 entries, randomly corrupt 20% of the data for the first experiment (i.e. swap target labels) and 50% for the
other one and then train the two algorithms `dtl` and `my-dtl` on the data and evaluate it on the uncorrupted version of
the data.

the names you for the `-a` flag are the ones you register the algorithms with in your code. By convention, the first
algorithm should be `dtl` here which the subsequent algorithms will be compared against. You can supply multiple
flags `-s`, `-d` and `-c` flags, in this case the number of experiment run will depend on all possible combinations for
the parameters, e.g. for `-c 0.2 -c 0.5 -s 5 -s 10` it will run 4 experiments, with 2 datasets of 2^5 and 2^10 examples,
corrupting 20% and 50% of each dataset.

The vigilant observer might find that we should in theory split training and evaluation data, but for simplicity we just
evaluate on the uncorrupted version. The difference is just that the final score is inflated by the performance on the
of uncorrupted data, which is the same for training and evaluation.

### Evaluate on less training examples

run

```bash
python evaluate_attributes.py eval-size --help
```

to understand what all the options do. Once you implemented a version of the algorithm for which you think your
algorithm performs better than the baseline developed in exercise 1, you can test it by running, e.g.

```bash
python evaluate_attributes.py eval-size -a dtl -a my-dtl -t 0.5 -t 0.7 -s 10 -d 10
```

The `-s` and `-d` parameters behave exactly like in the command above, `-t` splits the dataset in training evaluation
data according to the ratio (i.e. 50%:50% and 70%:30% here) trains on the first split and evaluates on the second split.
You might find that the baseline already performs reasonably well for most of the parameter space (e.g. when we have a
small size for the minimal consistent determination or when the training split is quite large). A reason for this might
be that the pseudo-random hypothesis is not particularly hard to learn. If you feel like increasing the complexity of
the hypothesis, you are free to do so, but again, don't forget to document the changes!

### Eval training duration

run

```bash
python evaluate_attributes.py eval-time --help
```

to understand what all the options do. Once you implemented a version of the algorithm for which you think that your
algorithm trains faster than the baseline developed in exercise 1, you can test it by running, e.g.

```bash
python evaluate_attributes.py eval-time -a dtl -a my-dtl -s 5 -s 7 -s 10
```

this will train the algorithms on {5,6,10}**2 examples and measure the average training time. Here, we don't compare for
statistical significance, as your gain should be noticeable by at least a couple of milliseconds (for the bigger
datasets). At the very least your gain should be outside the margin of uncertainty of the measurement (i.e. the number
behind the `+/-`.)

### In general

In order to get good marks here, you will need to show that your implementation is better for a good portion of the
parameter space. That means that your implementation should be better for multiple `-s` or -`d` and `-t` or `c`s
respectively, to show that your results are not just a fluke.

## Evaluate the Rule-learner

run

```bash
python evaluate_rules.py --help
```

to understand what all the options do. Once you implemented a version of the algorithm for which you think that your
algorithm is better than the foil baseline developed in exercise 2, you can test it by running, e.g.

```bash
python evaluate_rules.py -a my-foil -d tests/graph_small.json -k tests/graph_small.pl -t 'reachable(X,Y)' -r -e 0.8
```

This will run the algorithm my-foil on the dataset in `graph_small.json` and the corresponding knowledge
base `tests/graph_small.pl`, to learn the `reachable(X,Y)`
target, allowing to learn recursive rules (the `-r` flag) and splitting the dataset with a 80%:20% ratio for training
and evaluation. You can again supply multiple algorithms here, but given the fact that pyswip/swipl likes to crash, you
can run the experiments separately.

### Training data and knowledgebase

You can write your own dataset and the corresponding KB, to exemplify the improvements that you made to the baseline
foil algorithm. The dataset needs to have the following format:

```json
{
  "pos": [ // this is a list of all positive examples
    { // this is a single example
      "X": "someValue", // the keys in the example should correspond to the 
      "Y": "someOtherValue" // variables of the target predicate to learn
    } 
  ],
  "neg": [ // this is a list of all positive examples
    { // same as above
      "X": "someYetalue",
      "Y": "someAnotherOtherValue"
    }
  ]
}
```
The knowledge base should be normal a normal prolog file, interpretable by swipl.

You can make use of the ```generate_graphs.py``` to generate training data and
knowledge base describing [pseudo-random graphs](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model). You can run
```bash
python generate_graphs.py --help
```
to learn what the options are. For example, running
```bash
python generate_graphs.py abcde -n 15 -p 0.046
```
will generate a graph with 15 nodes and the probability of an edge between two
nodes being 4.6%. The edges will be saved in the 'abcde.pl' as the predicates
`connected`. Positive and negative examples for `reachable(X,Y)` relation will
be saved in `abcde.json`. You can adapt the script to generate examples for 
other interesting relations.

### In general

Here, we are not looking for statistical significance, but rather for conceptual improvements. Therefore, the difference
to the baseline should be noticeable, i.e. when evaluating, the baseline would get very low scores whereas your
implementation would get perfect score, or you can argue by comparing the learned hypotheses.
