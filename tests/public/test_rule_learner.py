import json
import re

import pytest

from learning.rule_learner import FOIL, HornClause, Conjunction, Disjunction, Expression, Literal, \
    is_represented_by, \
    Predicate
from learning.util import Dataset, Examples

ex_pos: Examples = [{"X": "elizabeth", "Y": "anne"},
                    {"X": "elizabeth", "Y": "andrew"},
                    {"X": "philip", "Y": "anne"},
                    {"X": "philip", "Y": "andrew"},
                    {"X": "anne", "Y": "peter"},
                    {"X": "anne", "Y": "zara"},
                    {"X": "mark", "Y": "peter"},
                    {"X": "mark", "Y": "zara"},
                    {"X": "andrew", "Y": "beatrice"},
                    {"X": "andrew", "Y": "eugenie"},
                    {"X": "sarah", "Y": "beatrice"},
                    {"X": "sarah", "Y": "eugenie"}]
ex_neg: Examples = [{"X": "anne", "Y": "eugenie"},
                    {"X": "beatrice", "Y": "eugenie"},
                    {"X": "mark", "Y": "elizabeth"},
                    {"X": "beatrice", "Y": "philip"}]

ex_pos_1 = [{'X': 'elizabeth', 'Y': 'peter'},
            {'X': 'elizabeth', 'Y': 'zara'},
            {'X': 'elizabeth', 'Y': 'beatrice'},
            {'X': 'elizabeth', 'Y': 'eugenie'},
            {'X': 'philip', 'Y': 'peter'},
            {'X': 'philip', 'Y': 'zara'},
            {'X': 'philip', 'Y': 'beatrice'},
            {'X': 'philip', 'Y': 'eugenie'}]

ex_neg_1 = [{'X': 'anne', 'Y': 'eugenie'},
            {'X': 'beatrice', 'Y': 'eugenie'},
            {'X': 'elizabeth', 'Y': 'andrew'},
            {'X': 'elizabeth', 'Y': 'anne'},
            {'X': 'elizabeth', 'Y': 'mark'},
            {'X': 'elizabeth', 'Y': 'sarah'},
            {'X': 'philip', 'Y': 'anne'},
            {'X': 'philip', 'Y': 'andrew'},
            {'X': 'anne', 'Y': 'peter'},
            {'X': 'anne', 'Y': 'zara'},
            {'X': 'mark', 'Y': 'peter'},
            {'X': 'mark', 'Y': 'zara'},
            {'X': 'andrew', 'Y': 'beatrice'},
            {'X': 'andrew', 'Y': 'eugenie'},
            {'X': 'sarah', 'Y': 'beatrice'},
            {'X': 'mark', 'Y': 'elizabeth'},
            {'X': 'beatrice', 'Y': 'philip'},
            {'X': 'peter', 'Y': 'andrew'},
            {'X': 'zara', 'Y': 'mark'},
            {'X': 'peter', 'Y': 'anne'},
            {'X': 'zara', 'Y': 'eugenie'}]


def test_overall_public(caplog):
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="parent(X,Y)")
    with FOIL(dataset=d, recursive=False) as foil:
        foil.fit()
        print(foil.hypothesis)
        assert isinstance(foil.hypothesis, Disjunction)
        assert HornClause(Literal.from_str("parent(X,Y)"),
                          Conjunction([Literal.from_str("father(X,Y)")])) in foil.hypothesis.expressions
        assert HornClause(Literal.from_str("parent(X,Y)"),
                          Conjunction([Literal.from_str("mother(X,Y)")])) in foil.hypothesis.expressions


def test_get_predicates_public():
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="parent(X,Y)")
    with FOIL(dataset=d) as f:
        assert set(f.get_predicates()) == {Predicate("male", 1), Predicate("female", 1), Predicate("mother", 2),
                                           Predicate("father", 2)}


def test_generate_candidates():
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="parent(X,Y)")
    with FOIL(dataset=d) as f:
        c = HornClause(Literal.from_str('parent(X,Y)'))
        candidates = list(f.generate_candidates(clause=c, predicates=[Predicate("mother", 2)]))
        assert len(candidates) == 8
        expected = {'mother(X,X)', 'mother(X,Y)', 'mother(Y,X)', 'mother(Y,Y)', 'mother(X,V_0)', 'mother(Y,V_0)',
                    'mother(V_0,X)', 'mother(V_0,Y)'}
        assert expected == set(str(c) for c in candidates)


def test_extend_example_public():
    d = Dataset(examples=ex_pos + ex_neg, kb='tests/test_kb.pl', target="parent(X,Y)")
    with FOIL(dataset=d) as f:
        result = list(f.extend_example({'X': "mark", 'Y': 'zara'}, new_expr=Literal.from_str('father(X, V_0)')))
        assert {'X': 'mark', 'Y': 'zara', 'V_0': 'zara'} in result
        assert {'X': 'mark', 'Y': 'zara', 'V_0': 'peter'} in result
        assert not list(f.extend_example({'X': "elizabeth", 'Y': 'andrew'}, new_expr=Literal.from_str('father(X, V_0)')))


def test_new_clause_public():
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="target(X,Y)")
    with FOIL(dataset=d) as f:
        new_clause = f.new_clause(positive_examples=[{"X": "sarah", "Y": "beatrice"},
                                                     {"X": "sarah", "Y": "eugenie"},
                                                     {"X": "anne", "Y": "peter"},
                                                     {"X": "anne", "Y": "zara"},
                                                     {"X": "elizabeth", "Y": "anne"},
                                                     {"X": "elizabeth", "Y": "andrew"}], negative_examples=ex_neg,
                                  predicates=f.get_predicates(), target=Literal.from_str('target(X,Y)'))

    assert isinstance(new_clause, HornClause)
    assert new_clause.head == Literal.from_str('target(X,Y)')
    assert len(new_clause.body.expressions) == 1
    assert new_clause.body.expressions[0] == Literal.from_str('mother(X,Y)')


def test_foil_ig_public():
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="target(X,Y)")
    with FOIL(dataset=d) as f:
        ig = f.foil_information_gain(Literal.from_str('mother(X,Y)'), ex_pos[:2], ex_neg[:2])
        assert ig == pytest.approx(2)
        ig2 = f.foil_information_gain(Literal.from_str('mother(X,V_0)'), ex_pos[:2], ex_neg[:2])
        assert ig2 == pytest.approx(0.83, rel=0.01)


def test_is_represented_by_public():
    assert is_represented_by({"X": 5, "Y": 6}, [{"X": 5, "Y": 6, "Z": 7}])
    assert not is_represented_by({"X": 5, "Y": 6}, [{"X": 5, "Y": 8, "Z": 7}])


def test_rule_covers_public():
    d = Dataset(positive_examples=ex_pos, negative_examples=ex_neg, kb='tests/test_kb.pl', target="target(X,Y)")
    with FOIL(dataset=d) as f:
        h = HornClause(head=Literal.from_str('parent(X,Y)'), body=Conjunction([Literal.from_str("mother(X,Y)")]))
        assert not f.covers(h, {"X": "mark", "Y": "peter"})


def test_overall_public_2():
    d = Dataset(positive_examples=ex_pos_1, negative_examples=ex_neg_1, kb='tests/test_kb2.pl',
                target="grandparent(X,Y)")
    with FOIL(dataset=d) as foil:
        foil.fit()
        print(foil.hypothesis)
        assert isinstance(foil.hypothesis, Disjunction)
        assert len(foil.hypothesis.expressions) == 1
        hypothesis = foil.hypothesis.expressions[0]
        assert isinstance(hypothesis, HornClause)
        assert hypothesis.head == Literal.from_str('grandparent(X,Y)')
        assert len(hypothesis.body.expressions) == 2
        string_repr_hyp = str(hypothesis)
        assert re.findall('parent\(X,V_\d+\)', string_repr_hyp)
        assert re.findall('parent\(V_\d+,Y\)', string_repr_hyp)


def test_recursive_public():
    with open("tests/graph_small.json") as f:
        data = json.load(f)

    d = Dataset(positive_examples=data['pos'], negative_examples=data['neg'], kb='tests/graph_small.pl',
                target='reachable(X,Y)')
    with FOIL(dataset=d, recursive=True) as f:
        f.fit()
        print(f.hypothesis)
        assert isinstance(f.hypothesis, Disjunction)
        horn_clauses = f.hypothesis.expressions
        bodies = [e.body for e in horn_clauses]
        assert any("connected(X,Y)" in str(b) for b in bodies)
        assert any(re.findall('reachable\(V_\d+,Y\)', str(b)) and re.findall('reachable\(X,V_\d+\)', str(b)) for b in bodies)
