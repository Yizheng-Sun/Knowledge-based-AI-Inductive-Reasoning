mother(anne, peter) .
mother(anne, zara) .
mother(sarah, beatrice) .
mother(sarah, eugenie) .
mother(elizabeth, anne) .
mother(elizabeth, andrew) .

father(mark, peter) .
father(mark, zara) .
father(andrew, beatrice) .
father(andrew, eugenie) .
father(philip, anne) .
father(philip, andrew) .

male(philip) .
male(mark) .
male(andrew) .
male(peter) .

female(elizabeth) .
female(anne) .
female(sarah) .
female(zara) .
female(beatrice) .
female(eugenie) .

parent(X,Y) :- mother(X,Y) .
parent(X,Y) :- father(X,Y) .