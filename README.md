# Knowledge-based-AI-Inductive-Reasoning
Algorithm code is in learning folder. This program uses learning decision tree algorithm and First-Order Inductive Learner (FOIL) algorithm to perform inductive reasoning. 

### Learning decision tree algorithm 
#### Code is in learning/attr_learner.py
Given a set of FOL formulas, we clausify them to transform them to clause normal form. In clause normal form, each literal can be seen as an attribute. We choose a goal atrribute among them. Learning decision tree algorithm can tell us the relation between our goal attribute and other attributes.

### FOIL algorithm
#### Code is in learning/rule_learner.py
The restriction of learning decision tree algorithm is that it can only learn local relation (relation between literals), it cannot learn relation between clauses. While FOIL algorithm can not only learn local relation, it can also learn global relation 
