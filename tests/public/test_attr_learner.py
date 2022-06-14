import random
import pytest

from learning.attr_learner import Node, Leaf, to_logic_expression, same_target, plurality_value, binary_entropy, \
    Disjunction, Conjunction, DecisionTreeLearner
from learning.generate import AttributeDatasetGenerator
from learning.util import Dataset


def test_tree_to_logic_expression_public():
    expected_1 = """ A0 = False AND A1 = Low AND A3 = True"""
    expected_2 = """A0 = False AND A1 = Low AND A3 = False AND A4 = False"""
    expected_3 = """A0 = False AND A1 = Mid AND A4 = True"""
    expected_4 = """A0 = False AND A1 = High AND A2 = Warm AND A4 = True"""
    expected_5 = """A0 = False AND A1 = High AND A2 = Warm AND A4 = False AND A5 = False"""

    tree = Node(attr_name="A0", branches={
        True: Leaf(False),
        False: Node(attr_name="A1", branches={
            "Low": Node(attr_name="A3", branches={
                True: Leaf(True),
                False: Node(attr_name="A4", branches={
                    True: Leaf(False),
                    False: Leaf(True)
                })
            }),
            "Mid": Node(attr_name="A4", branches={
                True: Leaf(True),
                False: Leaf(False)
            }),
            "High": Node(attr_name="A2", branches={
                "Cold": Leaf(False),
                "Warm": Node(attr_name="A4", branches={
                    True: Leaf(True),
                    False: Node(attr_name="A5", branches={
                        True: Leaf(False),
                        False: Leaf(True)
                    })
                })
            })
        })
    })
    logic_expression = to_logic_expression(tree)
    assert isinstance(logic_expression, Disjunction)
    assert len(logic_expression.conjunctions) == 5
    for e in [expected_1, expected_2, expected_3, expected_4, expected_5]:
        assert e in str(logic_expression)


def test_same_target_public():
    assert same_target([{"A": True, 'target': True}, {"A": False, 'target': True}])
    assert not same_target([{"A": True, 'target': True}, {"A": False, 'target': False}])


def test_plurality_value_public():
    assert plurality_value([{"A": True, 'target': True}, {"A": False, 'target': True}])
    assert not plurality_value([{"A": True, 'target': False}, {"A": False, 'target': False}])
    assert plurality_value([{"A": True, 'target': True}, {"A": False, 'target': True}, {"A": False, 'target': True}])


def test_binary_entropy_public():
    ds = [{f"A{i}": random.choice([True, False]) for i in range(4)} for _ in range(15)]
    for d in ds[:10]:
        d['target'] = True
    for d in ds[10:]:
        d['target'] = False
    assert binary_entropy(ds) == pytest.approx(0.918, rel=0.005)


def test_information_gain_public():
    pos_targets = 4 * [False] + 5 * [True]
    pos = [{'A': False, 'target': pos_targets.pop()} for _ in range(9)]
    neg = [{'A': True, 'target': False} for _ in range(6)]
    a = DecisionTreeLearner(Dataset(pos + neg))
    assert a.information_gain(pos + neg, 'A') == pytest.approx(0.324, rel=0.005)


def test_overall_public():
    c1 = Conjunction({'A1': False, 'A2': True})
    c2 = Conjunction({'A4': True})
    d1 = Disjunction([c1, c2])
    attr_values = {"A1": [True, False], "A2": [True, False], "A3": [True, False], "A4": [True, False]}
    gen = AttributeDatasetGenerator(attr_values, d1)
    ds = Dataset(gen.generate_all())
    algo = DecisionTreeLearner(ds)
    algo.fit()
    print(algo.hypothesis)
    for d in ds:
        assert algo.hypothesis(d) == d1(d)


def test_get_most_important_attribute_public():
    c1 = Conjunction({"A1": False})
    c2 = Conjunction({"A2": True})
    d1 = Disjunction([c1, c2])
    attr_values = {"A1": [True, False], "A2": [True, False], "A3": [True, False]}
    gen = AttributeDatasetGenerator(attr_values, d1)
    examples = gen.generate_all()
    examples.remove({"A1": False, "A2": False, "A3": True, "target": True})
    ds = Dataset(examples)
    algo = DecisionTreeLearner(ds)
    assert algo.get_most_important_attribute(ds.attributes, examples) == "A2"
