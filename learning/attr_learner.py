from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from operator import pos
from re import T
from typing import Any, List, Dict

from learning.util import Algorithm, AlgorithmRegistry

Example = Dict[str, Any]
Examples = List[Example]

from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class AttrLogicExpression(ABC):
    """
    Abstract base class representing a logic expression.
    """
    ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass(frozen=True)
class Conjunction(AttrLogicExpression):
    """
    A configuration of attribute names and the values the attributes should take for this conjunction to evaluate
    to true.

    `attribute_confs` is a map from attribute names to their values.
    """
    attribute_confs: Dict[str, Any]

    def __post_init__(self):
        assert 'target' not in self.attribute_confs, "Nice try, but 'target' cannot be part of the hypothesis."

    def __call__(self, example: Example):
        """
        Evaluates whether the conjunction applies to an example or not. Returns true if it does, false otherwise.


        Args:
            example: Example to check if the conjunction applies.

        Returns:
            True if the values of *all* attributes mentioned in the conjunction and appearing in example are equal,
            false otherwise.


        """
        return all(self.attribute_confs[k] == example[k] for k in set(self.attribute_confs).intersection(example))

    def __repr__(self):
        return " AND ".join(f"{k} = {v}" for k, v in self.attribute_confs.items())


@dataclass(frozen=True)
class Disjunction(AttrLogicExpression):
    """
    Disjunction of conjunctions.
    """
    conjunctions: List[Conjunction]

    def __call__(self, example: Example):
        """
        Evaluates whether the disjunction applies to a given example.

        Args:
            example: Example to check if the disjunction applies.

        Returns: True if any of its conjunctions returns true, and false if none evaluates to true.

        """
        return any(c(example) for c in self.conjunctions)

    def __repr__(self):
        return " " + "\nOR\n ".join(f"{v}" for v in self.conjunctions)


class Tree(ABC):
    """
    This is an abstract base class representing a leaf or a node in a tree.
    """
    ...


@dataclass
class Leaf(Tree):
    """
    This is a leaf in the tree. It's value is the (binary) classification, either True or False.
    """
    target: bool


@dataclass
class Node(Tree):
    """
    This is a node in the tree. It contains the attribute `attr_name` which the node is splitting on and a dictionary
    `branches` that represents the children of the node and maps from attribute values to their corresponding subtrees.
    """
    attr_name: str
    branches: Dict[Any, Tree] = field(default_factory=dict)


def same_target(examples: Examples) -> bool:
    """
    This function checks whether the examples all have the same target.

    Args:
        examples: Observations to check

    Returns: Whether the examples all have the same target.
    """

    if (all(examples[i]['target']==examples[0]['target'] for i in range(len(examples)))):
        return True
    else:
        return False

def dominant(examples: Examples):
    pos=0
    neg=0
    for e in examples:
        if (e['target']):
            pos+=1
        else:
            neg+=1
    if pos/(pos+neg) >= 0.8:
        return True
    if neg/(pos+neg) >= 0.8:
        return False
    
    return "no_dominant"

def plurality_value(examples: Examples) -> bool:
    """
    This function returns whether there are more positively or negatively classified examples in the dataset.
    Args:
        examples: Examples to check.

    Returns: True if more examples classified as positive, False otherwise.

    """
    pos_ex = 0
    neg_ex = 0
    # if target is true, then it's a positive example
    # otherwise it's a negtive example
    for e in examples:
        if e['target']:
            pos_ex += 1
        else:
            neg_ex += 1
    # Compare the number of positive and negtive examples
    # Return the plurality
    if pos_ex >= neg_ex:
        return True
    else:
        return False


def binary_entropy(examples: Examples) -> float:
    """
    Calculates the binary (shannon) entropy of a dataset regarding its classification.
    Args:
        examples: Dataset to calculate the shannon entropy.

    Returns: The shannon entropy of the classification of the dataset.

    """
    false_target = 0
    true_target = 0

    # if the target of this example is true
    # then this example is a positive example
    # if the target of this example is false
    # then this example is a negtive example
    for e in examples: 
        if e['target']:
            true_target+=1
        else:
            false_target+=1

    if (false_target+true_target != 0):
        # add 0.0000001 here to avoid division by 0
        pr_t = true_target/(false_target+true_target)+0.0000001
        pr_f = false_target/(true_target+false_target)+0.0000001
    else:
        pr_t = 0.0000001
        pf_f = 0.0000001
    entropy = - (pr_t * math.log(pr_t,2) + pr_f * math.log(pr_f,2))
    return entropy
    

def to_logic_expression(tree: Tree) -> AttrLogicExpression:
    """
    Converts a Decision tree to its equivalent logic expression.
    Args:
        tree: Tree to convert.

    Returns: The corresponding logic expression consisting of attribute values, conjunctions and disjunctions.

    """
    conjunctions_list = []
    conjunction = Conjunction(dict())
    # this function will get all the true branches of the tree
    # and convert each true branch to a conjunction
    getAllConjunction(tree, conjunction, conjunctions_list)
    disjunction = Disjunction(conjunctions_list)
    return disjunction

# This function returns all the branches of a tree
def getAllConjunction(tree: Tree, conjunction: Conjunction, ls):
    # if the tree is a leaf ,that means we have reaches the bottom of the tree
    # if the target of this branch is True, then we convert this branch to a conjuction
    # if the target of this branch is not a Tree, then we ignore this branch
    if type(tree) == Leaf:
        if(tree.target):
            dict = conjunction.attribute_confs.copy()
            con = Conjunction(dict)
            ls.append(con)
    else:
        # for each possible value of a tree, we go the branch with this value
        # Then recursively call this function to go to next layer of the tree
        for b in tree.branches:
            dict = conjunction.attribute_confs.copy()
            con1 = Conjunction(dict)
            con1.attribute_confs[tree.attr_name] = b 
            getAllConjunction(tree.branches[b], con1, ls)

@AlgorithmRegistry.register("dtl")
class DecisionTreeLearner(Algorithm):
    """
    This is the decision tree learning algorithm.
    """

    def find_hypothesis(self) -> AttrLogicExpression:
        tree = self.decision_tree_learning(examples=self.dataset.examples, attributes=self.dataset.attributes, parent_examples=[])
        return to_logic_expression(tree)

    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        """
        This is the main function that learns a decision tree given a list of example and attributes.
        Args:
            examples: The training dataset to induce the tree from.
            attributes: Attributes of the examples.
            parent_examples: Examples from previous step.

        Returns: A decision tree induced from the given dataset.
        """
        # if there are no examples left, that means we do not have sufficient examples,
        # the best we can do is to return the current best target we have,
        # which is the plurality value in the parent examples 
        if len(examples) == 0:
            leaf = Leaf()
            leaf.target = plurality_value(parent_examples)
            return leaf
        # if all the examples have the same classes
        # this is ideal base case. That means we can find a suitable hypothesis for all the examples
        elif same_target(examples):
            leaf = Leaf(examples[0]['target'])
            return leaf

        # if the attribute of an example is example, that means these examples are noise
        # they cannot give us any further information, but the best we can do is to return the current best target we have.
        # which is the plurality of these examples
        elif len(attributes) == 0:
            leaf = Leaf()
            leaf.target = plurality_value(examples)
            return leaf
        else:
            # this is the step case,
            # in each recursive call, we get the attribute with the largest information gain
            attr = self.get_most_important_attribute(attributes, examples)
            node = Node(attr)
            attr_ls = [attr]
            # then remove this chose attribute from all the attribute
            # because we will split branches according to this attribute
            new_attrs = [x for x in attributes if x not in attr_ls]
            vals = set() 
            for e in examples:
                vals.add(e[attr])
            # split branches according to attribute's value
            for v in vals:
                new_ex = [e for e in examples if e[attr] == v]
                subTree = self.decision_tree_learning(new_ex, new_attrs, examples)
                node.branches[v] = subTree
            return node

    def get_most_important_attribute(self, attributes: List[str], examples: Examples) -> str:
        """
        Returns the most important attribute according to the information gain measure.
        Args:
            attributes: The attributes to choose the most important attribute from.
            examples: Dataset from which the most important attribute is to be inferred.

        Returns: The most informative attribute according to the dataset.

        """
        ig = []
        # add information gain value of each attribute to a list
        for a in attributes:
            ig.append(self.information_gain(examples, a))
        # get the attribute with the highest information gain
        index_max = max(range(len(ig)), key=ig.__getitem__)
        return attributes[index_max]
        


    def information_gain(self, examples: Examples, attribute: str) -> float:
        """
        This method calculates the information gain (as presented in the lecture)
        of an attribute according to given observations.

        Args:
            examples: Dataset to infer the information gain from.
            attribute: Attribute to infer the information gain for.

        Returns: The information gain of the given attribute according to the given observations.

        """
        attr_pos = 0
        attr_neg = 0
        attr_pos_tar_pos = 0
        attr_pos_tar_neg = 0
        attr_neg_tar_pos = 0
        attr_neg_tar_neg = 0

        for e in examples:
            # if this  
            if e[attribute] :
                attr_pos += 1
                if e['target']:
                    attr_pos_tar_pos += 1
                else:
                    attr_pos_tar_neg += 1
            else:
                attr_neg += 1
                if e['target']:
                    attr_neg_tar_pos += 1
                else:
                    attr_neg_tar_neg += 1
        pr_attr_pos = attr_pos/(attr_pos + attr_neg)
        pr_attr_neg = attr_neg/(attr_pos + attr_neg)
        pr_attr_pos_tar_pos = (attr_pos_tar_pos/(attr_pos_tar_pos+attr_pos_tar_neg))+0.0000001
        pr_attr_pos_tar_neg = (attr_pos_tar_neg/(attr_pos_tar_pos+attr_pos_tar_neg))+0.0000001
        pr_attr_neg_tar_pos = (attr_neg_tar_pos/(attr_neg_tar_pos+attr_neg_tar_neg))+0.0000001
        pr_attr_neg_tar_neg = (attr_neg_tar_neg/(attr_neg_tar_pos+attr_neg_tar_neg))+0.0000001
        entropy_attr = pr_attr_pos*(-(pr_attr_pos_tar_pos * math.log(pr_attr_pos_tar_pos,2) + pr_attr_pos_tar_neg * math.log(pr_attr_pos_tar_neg,2))) + pr_attr_neg*(-(pr_attr_neg_tar_pos * math.log(pr_attr_neg_tar_pos,2) + pr_attr_neg_tar_neg * math.log(pr_attr_neg_tar_neg,2)))
        entropy_tar = binary_entropy(examples)
        information_gain = entropy_tar - entropy_attr
        return information_gain

@AlgorithmRegistry.register("my-dtl")
class MyDecisionTreeLearner(DecisionTreeLearner):
    def decision_tree_learning(self, examples: Examples, attributes: List[str], parent_examples: Examples) -> Tree:
        # if there are no examples left, that means we do not have sufficient examples,
        # the best we can do is to return the current best target we have,
        # which is the plurality value in the parent examples 
        if len(examples) == 0:
            leaf = Leaf()
            leaf.target = plurality_value(parent_examples)
            return leaf
        # if all the examples have the same classes
        # this is ideal base case. That means we can find a suitable hypothesis for all the examples
        elif same_target(examples):
            leaf = Leaf(examples[0]['target'])
            return leaf
        
        # improvement
        # We don't wait until all the examples have the same target
        # if there are more than 90% examples have the same target, we return this target
        # This is cut off some search space
        elif dominant(examples) != "no_dominant":
            if (dominant(examples)):
                return Leaf(True)
            else:
                return Leaf(False)

        # if the attribute of an example is example, that means these examples are noise
        # they cannot give us any further information, but the best we can do is to return the current best target we have.
        # which is the plurality of these examples
        elif len(attributes) == 0:
            leaf = Leaf()
            leaf.target = plurality_value(examples)
            return leaf
        else:
            # this is the step case,
            # in each recursive call, we get the attribute with the largest information gain
            attr = self.get_most_important_attribute(attributes, examples)
            node = Node(attr)
            attr_ls = [attr]
            # then remove this chose attribute from all the attribute
            # because we will split branches according to this attribute
            new_attrs = [x for x in attributes if x not in attr_ls]
            vals = set() 
            for e in examples:
                vals.add(e[attr])
            # split branches according to attribute's value
            for v in vals:
                new_ex = [e for e in examples if e[attr] == v]
                subTree = self.decision_tree_learning(new_ex, new_attrs, examples)
                node.branches[v] = subTree
            return node
