from calendar import c
import itertools
import math
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, astuple
from typing import List, Generator, Union
from pyswip import Functor, Variable, Query, call, Prolog

from learning.util import Algorithm, Examples, Dataset, Example, AlgorithmRegistry

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class Predicate:
    """
    Object representation of a predicate. Contains `name` which is the name of the predicate and its `arity`.
    """
    name: str
    arity: int

    def __post_init__(self):
        assert self.name[0].islower()


@dataclass(frozen=True)
class Expression:
    """
    Abstract base class representing a valid logical statement.
    """
    ...


@dataclass(frozen=True)
class Literal(Expression):
    """
    Literal: A Predicate with instantiated values for its arguments, which can be either variables or atomic values.

    Converting the literal to string will yield its syntactically valid prolog representation.
    """
    predicate: Predicate = field(hash=True)
    arguments: List[Union['Expression', str]] = field(hash=True)

    def __post_init__(self):
        """
        Make sure that the number of arguments corresponds to the predicate's arity.

        """
        assert (len(self.arguments) == self.predicate.arity,
                f"Number of arguments {len(self.arguments)} not "
                f"equal to the arity of the predicate {self.predicate.arity}")

    def __repr__(self):
        """
        Prolog representation.

        Returns: A syntactically valid prolog representation of the literal.

        """
        return f"{self.predicate.name}({','.join(str(a) for a in self.arguments)})"

    @classmethod
    def from_str(cls, string):
        """
        Generates a python object from a syntactically valid prolog representation.
        Args:
            string: Prolog representation of the literal.

        Returns: `Literal` object equivalent to the prolog representation.

        """
        predicate = get_predicate(string)
        args = get_args(string)
        return Literal(predicate, args)


def get_predicate(text: str) -> Predicate:
    """
    Returns the name and arity of a predicate from a syntactically valid prolog representation.
    Args:
        text: Text to extract the predicate from.

    Returns: Object of `Predicate` class with its corresponding name and arity.

    """
    text = str(text)
    name = text[:text.find("(")].strip()
    arity = len(re.findall("Variable", text))
    if arity == 0:
        arity = len(re.findall(",", text)) + 1
    return Predicate(name, arity)


@dataclass(frozen=True)
class Disjunction(Expression):
    """
    Represents a disjunction of horn clauses which is initially empty.
    """
    expressions: List['HornClause'] = field(default_factory=list)

    def generalize(self, expression: 'HornClause'):
        """
        Adds another horn clause to the disjunction.
        Args:
            expression: Horn clause to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the horn clauses.

        Since there is no real disjunction in prolog, this is just a set of the expressions as separate statements.
        Returns:
            syntactically valid prolog representation of the contained horn clauses.

        """
        return " .\n".join(repr(e) for e in self.expressions) + ' .'


@dataclass(frozen=True)
class Conjunction(Expression):
    """
    Represents a conjunction of literals which is initially empty.
    """
    expressions: List[Expression] = field(default_factory=list)

    def specialize(self, expression: Expression):
        """
        Adds another literal to the conjunction.
        Args:
            expression: literal to add

        """
        self.expressions.append(expression)

    def __repr__(self):
        """
        Returns a syntactically valid prolog representation of the conjunction of the literals.

        Returns:
            syntactically valid prolog representation of the conjunction (comma-separated).

        """
        return " , ".join(repr(e) for e in self.expressions)


@dataclass(frozen=True)
class HornClause(Expression):
    """
    Represents a horn clause with a literal as `head` and a conjunction as `body`.
    """
    head: Expression
    body: Conjunction = field(default_factory=lambda: Conjunction())

    def get_vars(self):
        """
        Returns all variables appearing in the horn clause.

        Returns: All variables in the horn clause, according to prolog syntax, where variables are capitalised.

        """
        return re.findall(r"(?:[^\w])([A-Z]\w*)", str(self))

    def __repr__(self):
        """
        Converts to a syntactically valid prolog representation.

        Returns:
            Syntactically valid prolog representation of a horn clause in the form of
            ``head :- literal_1 , literal_2 , ... , literal_n``
            for all literals in the body.
        """
        return f"{str(self.head)} :- {' , '.join(str(e) for e in self.body.expressions)}"


def get_args(text: str) -> List[str]:
    """
    Returns the arguments of a text that is assumed to be a single literal in prolog representation.

    Args:
        text: Text to extract the arguments from. Must be valid prolog representation of a single literal.

    Returns:
        All arguments that appear in that literal.

    """
    return [x.strip() for x in re.findall(r"\(.*\)", str(text))[0][1:-1].split(",")]


@AlgorithmRegistry.register('foil')
@contextmanager
def FOIL(dataset: Dataset, recursive=False):
    f = _FOIL(dataset, recursive)
    try:
        yield f
    finally:
        f.abolish()


class _FOIL(Algorithm):
    prolog: Prolog
    count = 0
    recursive_predicate = []

    def __init__(self, dataset: Dataset, recursive=False):
        super().__init__(dataset)
        logger.info("Creating prolog...")
        self.prolog = Prolog()

        self.recursive = recursive

        if dataset.kb:
            logger.debug(f"Consulting {self.dataset.kb}")
            self.prolog.consult(self.dataset.kb)

    def abolish(self):
        for p, a in (astuple(a) for a in self.get_predicates()):
            self.prolog.query(f"abolish({p}/{a})")

    def predict(self, example: Example) -> bool:
        return any(self.covers(clause=c, example=example) for c in self.hypothesis.expressions)
        
        
    def get_predicates(self) -> List[Predicate]:
        """
        This method returns all (relevant) predicates from the knowledge base.

        Returns:
            all currently known predicates in the knowledge base that was loaded from the file corresponding to the
            dataset.

        """
        p = Variable()
        predicates1 = []
        predicates2 = []
        predicates3 = []
        predicates4 = []
        predicates_all = []
        # query with "defined" as the second parameter for predicate_property gives all the predicates in knowledge base
        # all the other queries can remove the irrelevant predicates from the knowledge base
        q1 = list(self.prolog.query("predicate_property(P, multifile)"))
        q2 = list(self.prolog.query("predicate_property(P, defined)"))
        q3 = list(self.prolog.query("predicate_property(P, built_in)"))
        q4 = list(self.prolog.query("predicate_property(P, thread_local)"))
        q5 = list(self.prolog.query("predicate_property(P, dynamic)"))
        for pre in q1:
            predicates1.append(get_predicate(pre['P']))
        for pre in q2:
            predicates_all.append(get_predicate(pre['P']))
        for pre in q3:
            predicates2.append(get_predicate(pre['P']))
        for pre in q4:
            predicates3.append(get_predicate(pre['P']))
        for pre in q5:
            predicates4.append(get_predicate(pre['P']))
        # remove all the irrelavant predicates from all predicates, we can get the relavant predicates
        predicates = [x for x in predicates_all if x not in predicates1 and x not in predicates2 and x not in predicates3 and x not in predicates4] 
        return predicates

    def find_hypothesis(self) -> Disjunction:
        """
        Initiates the FOIL algorithm and returns the final disjunction from the list that is returned by
        `FOIL.foil`.

        Returns: Disjunction of horn clauses that represent the learned target relation.

        """
        positive_examples = self.dataset.positive_examples
        negative_examples = self.dataset.negative_examples

        target = Literal.from_str(self.dataset.target)

        predicates = self.get_predicates()
        assert predicates

        clauses = self.foil(positive_examples, negative_examples, predicates, target)
        return Disjunction(clauses)

    def foil(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
             target: Literal) -> List[HornClause]:
        """
        Learns a list of horn clauses from a set of positive and negative examples which as a disjunction
        represent the hypothesis inferred from the dataset.

        This method is the outer loop of the foil algorithm.

        Args:
            positive_examples: Positive examples for the target relation to be learned.
            negative_examples: Negative examples for the target relation to be learned.
            predicates: Predicates that are allowed in the bodies of the horn clauses.
            target: Signature of the target relation to be learned

        Returns:
            A list of horn clauses that as a disjunction cover all positive and none of the negative examples.

        """
        if self.recursive:
            tar_dum = Literal(target.predicate, ["dummy"]*target.predicate.arity)
            self.prolog.assertz(tar_dum.__repr__())
            predicates.append(target.predicate)
        clauses = []
        count = 0 # count counts for outter iteration number
        # when we have iterates for more than 20 times, we stop and return the current hypothesis
        while (len(positive_examples)>0 and count < 20):
            clause = self.new_clause(positive_examples, negative_examples, predicates, target)
            # get all the examples that are covered by our hypothesis,
            # this is for recursive relation
            covers_examples = [e for e in positive_examples if self.covers(clause,e)]
            positive_examples = [e for e in positive_examples if not self.covers(clause,e)]

            # works for recuresive relation
            # if there are examples coverd by our hypothesis
            # add these examples with target predicate to knowledge base
            # add target predicate to predicates list
            if self.recursive:
                arguments = []
                recursive_predicate = 0
                print(covers_examples)
                for ce in covers_examples:
                    for attr in ce:
                        arguments.append(ce[attr])
                    # make new literals, literal's predicate is target's predict
                    # literal's content is an example's content
                    recuresive_literal = Literal(target.predicate, arguments)
                    recusive_prolog = recuresive_literal.__repr__()
                    self.prolog.assertz(recusive_prolog)
                    recuresive_predicate = Predicate(recuresive_literal.predicate.name,len(arguments))
                    arguments=[]

            # add new hypothesis
            body_copy = clause.body.expressions.copy()
            conj = Conjunction(body_copy)
            new_horn = HornClause(target,conj)
            clauses.append(new_horn)
            count+=1
        return clauses

    def covers(self, clause: HornClause, example: Example) -> bool:
        """
        This method checks whether an example is covered by a given horn clause under the current knowledge base.
        Args:
            clause: The clause to check whether it covers the examples.
            example: The examples to check whether it is covered by the clause.

        Returns:
            True if covered, False otherwise

        """
        # make a prolog query by the body of the clause
        # predicate of the query is the clause's body 
        # content of the query is the example
        body = clause.body.expressions
        for i in body:
            args = i.arguments
            ex = ""
            # check whether the Variable name in clause is also in example
            # if not, a new unique variable is introduced
            for arg in args:
                try:
                    x = example[arg]
                except KeyError:
                    x = self.unique_var()
                finally:
                    ex += x+","
            ex = ex[:len(ex)-1]    
            predicate = get_predicate(i).name
            q = self.prolog.query(str(predicate)+"("+ex+")")
            # if we can get a query result, that means our example is covered by the clause
            if len(list(q))>0:
                continue
            else:
                return False
        return True

    def new_clause(self, positive_examples: Examples, negative_examples: Examples, predicates: List[Predicate],
                   target: Literal) -> HornClause:
        """
        This method generates a new horn clause from a dataset of positive and negative examples, a target and a
        list of allowed predicates to be used in the horn clause body.

        This corresponds to the inner loop of the foil algorithm.

        Args:
            positive_examples: Positive examples of the dataset to learn the clause from.
            negative_examples: Negative examples of the dataset to learn the clause from.
            predicates: List of predicates that can be used in the clause body.
            target: Head of the clause to learn.

        Returns:
            A horn clause that covers some part of the positive examples and does not contradict any of the
            negative examples.

        """
        h_clause = HornClause(target)
        count = 0 # count counts for inner iteration number
        # when we have iterates for more than 20 times, we stop and return the current clause
        while len(negative_examples) > 0 and count < 20:
            count+=1
            candidates = self.generate_candidates(h_clause, predicates)
            # We pick the candidate with the highest information gain
            lit = self.get_next_literal(candidates, positive_examples, negative_examples)
            h_clause.body.expressions.append(lit)

            # Update positive_examples and negtive examples by our new candidate
            new_pos = []
            new_neg = []
            for pe in positive_examples:
                new_pos+=self.extend_example(pe, lit)
            for ne in negative_examples:
                new_neg+=self.extend_example(ne, lit)
            positive_examples = new_pos
            negative_examples = new_neg
        return h_clause

    def get_next_literal(self, candidates: List[Expression], pos_ex: Examples, neg_ex: Examples) -> Expression:
        """
        Returns the next literal with the highest information gain as computed from a given dataset of positive and
        negative examples.
        Args:
            candidates: Candidates to choose the one with the highest information gain from.
            pos_ex: Positive examples of the dataset to infer the information gain from.
            neg_ex: Negative examples of the dataset to infer the information gain from.

        Returns:
            the next literal with the highest information gain as computed
            from a given dataset of positive and negative examples.

        """
        ig_ls = []
        # calculate the information gain of each candidate and add them to a list
        # print(candidates)
        for c in range(len(candidates)):
            ig = self.foil_information_gain(candidates[c], pos_ex, neg_ex)
            ig_ls.append(ig)
        # print(ig_ls)
        # We choose the candidate with the higheset information gain
        index_max = max(range(len(ig_ls)), key=ig_ls.__getitem__)
        lit = candidates[index_max]
        return lit

    def foil_information_gain(self, candidate: Expression, pos_ex: Examples, neg_ex: Examples) -> float:
        """
        This method calculates the information gain (as presented in the lecture) of an expression according
           to given positive and negative examples observations.

        Args:
               candidate: Attribute to infer the information gain for.
               pos_ex: Positive examples to infer the information gain from.
               neg_ex: Negative examples to infer the information gain from.

        Returns: The information gain of the given attribute according to the given observations.

        """
        ex_pos_new = []
        ex_neg_new = []
        t = 0
        # Update positive_examples and negative_examples by new candidate
        for e in pos_ex:
            ls = self.extend_example(e, candidate)
            ex_pos_new += ls
        for e in neg_ex:
            ls = self.extend_example(e, candidate)
            ex_neg_new += ls

        # t is the number of Examples covered by hypothesis and by specialised hypothesis
        for e in pos_ex:
            if is_represented_by(e, ex_pos_new):
                t+=1

        # add 0.0000001 to each value to avoid 0 devision
        p_1 = len(ex_pos_new)+0.00000001
        n_1 = len(ex_neg_new)+0.00000001
        p_0 = len(pos_ex)+0.00000001
        n_0 = len(neg_ex)+0.00000001
        ig = t*(math.log(p_1/(p_1+n_1),2) - math.log(p_0/(p_0+n_0),2))
        return ig

    def generate_candidates(self, clause: HornClause, predicates: List[Predicate]) -> Generator[Expression, None, None]:
        """
        This method generates all reasonable (as discussed in the lecture) specialisations of a horn clause
        given a list of allowed predicates.

        Args:
            clause: The clause to calculate possible specialisations for.
            predicates: Allowed predicate vocabulary to specialise the clause.

        Returns:
            All expressions that could be a reasonable specialisation of `clause`.

        """
        head = clause.head
        body = clause.body
        unique_var = self.unique_var()
        variables2 = []
        variables2.append(unique_var)
        all_expressions = []

        # Get all the variables that are already in the rule already (head or body) 
        # And add them to our candidate variables
        for arg in head.arguments:
            variables2.append(arg)
        for i in body.expressions:
            for arg in i.arguments:
                variables2.append(arg)
        
        # for each available predicates, create a literal with at least one variable from
        # the variables that are already in the rule
        for x in variables2:
            for y in variables2:
                for p in predicates:
                    if p.arity == 2:
                        # check if there are at least one variable that are already in the rule
                        if x != unique_var or y!=unique_var:
                            Lit = Literal(p,[x,y])
                            all_expressions.append(Lit)
                    # if arity is one, then the variable cannot be a newly introduced variable    
                    if p.arity == 1:
                        if x != unique_var:
                            Lit = Literal(p,[x])
                            all_expressions.append(Lit)
        return all_expressions

    def extend_example(self, example: Example, new_expr: Expression) -> Generator[Example, None, None]:
        """
        This method extends an example with all possible substitutions subject to a given expression and the current
        knowledge base.
        Args:
            example: Example to extend.
            new_expr: Expression to extend the example with.

        Returns:
            A generator that yields all possible substitutions for a given example an an expression.

        """
        extend_examples = []
        # make a prolog query with our new introduced rule
        new_expr_prolog = new_expr.__repr__()
        q = self.prolog.query(new_expr_prolog)
        for res in list(q):
            add = True
            for e in example:
                # if query result and example have the same Variable
                # if they are the same, we check next Variable
                # if they are differnt, we ignore this query
                for k2 in res:
                    if e == k2:
                        if example[e] != res[k2]:
                            add = False
            # if all the variables that are both in query result and example are the same
            # we add this query result to extended examples
            if(add):
                ex = example.copy()
                for r in res:
                    ex[r] = res[r]
                
                # if this example is already in extended examples, we ignore it
                same = False
                for x in extend_examples:
                    if x == ex:
                        same = True
                if not same:
                    extend_examples.append(ex)
            add = True
        # print(extend_examples)
        return extend_examples
        

    def unique_var(self) -> str:
        """
        Returns the next uniquely numbered variable to be used.

        Returns:
            the next uniquely named variable in the following format: `V_i` where `i` is a number.

        """
        # count is a public variables, every time we create a new unique variable
        # count+=1
        var = "V_"+str(self.count)
        self.count+=1
        return var

        


def is_represented_by(example: Example, examples: Examples) -> bool:
    """
    Checks whether a given example is represented by a list of examples.
    Args:
        example: Example to check whether it's represented.
        examples: Examples to check whether they represent the example.

    Returns:
        True, if for some `e` in `examples` for all variables (keys except target) in `example`,
        the values are equal (potential additional variables in `e` do not need to be considered). False otherwise.

    """
    # check each variable except 'target' of each example in `examples`,
    # with example to see if they are the same
    for e in examples:
        same = True
        for attr in example:
            if example[attr] != e[attr] and attr != 'target':
                same = False
        if same:
            return True
        else:
            continue
    return False
