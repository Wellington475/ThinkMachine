# ThinkMachine [![Build Status](https://travis-ci.org/Wellington475/ThinkMachine.svg?branch=master)](https://travis-ci.org/Wellington475/ThinkMachine)
Module for tools and algorithm to Data Science.
____
ThinkMachine is a set of tools and algorithms to facilitate the work of common data area Sciences tasks. It is based on a number of studies related to data science was by own creator.

#### Linear classifier:

Perceptron is an algorithm based on neural networks, heavily used for binary classification. He could solve any linearly solvable problem.

```python
from thinkmachine import PerceptronLinear
# OR      inputs   expected
dataset = [([1, 1],   1),
       ([1, 0],   1),
       ([0, 1],   1),
       ([0, 0],   0)]

p = PerceptronLinear(learning_rate=0.5) 
p.train(dataset)
print("Expected: "+str(dataset[0][1])+", Result: "+str(p.perceive([1, 1])))
```
#### Output:
```
Expected: 1, Result: 1
```

### Graph of errors
```python
from thinkmachine import PerceptronLinear
import matplotlib.pyplot as plt

# OR      inputs   expected
dataset = [([1, 1],   1),
       ([1, 0],   1),
       ([0, 1],   1),
       ([0, 0],   0)]

p = PerceptronLinear(learning_rate=0.5) 
p.train(dataset)

plt.plot(range(1, len(p.errors)+1), p.errors, marker='*')
plt.xlabel('Interações')
plt.ylabel('Niveis de erro')
plt.show()

```

#### Decision Tree:

```python

from thinkmachine import ID3

id3 = ID3()
id3.load_sample()
data = []
tree = None

id3.load_sample()

file = open('data_set/sample.csv')

for line in file:
  line = line.strip("\r\n")
  data.append(line.split(','))

attributes = data[0]
data.remove(attributes)
tree = id3.buildtree(data)
id3.printtree(tree)
# print: structure of decision tree
```

#### Output:

```
Column[0]: rain?
 True -> Decision: yes
 False -> Column[2]: high?
   True -> Column[3]: weak?
     True -> Column[0]: sunny?
       True -> Column[1]: pleasant?
         True -> Decision: no
         False -> Column[1]: hot?
           True -> Decision: no
           False -> Decision: yes
       False -> Decision: yes
     False -> Decision: no
   False -> Column[1]: pleasant?
     True -> Column[0]: sunny?
       True -> Decision: yes
       False -> Decision: no
     False -> Decision: yes
```



### Classify:

```python
from thinkmachine import ID3

# Col:    weather   temperature   humidity   wind       decision    
# index:  0         1             2          3          4
data = [['rain',    'cold',       'high',   'weak',     'yes'],
        ['sunny',   'hot',        'high',   'weak',     'no'],
        ['clouds',  'pleasant',   'high',   'weak',     'yes'],
        ['clouds',  'pleasant',   'normal', 'strong',   'no'],
        ['sunny',   'cold',       'high',   'weak',     'yes'],
        ['sunny',   'pleasant',   'high',   'weak',     'no'],
        ['clouds',  'cold',       'normal', 'strong',   'yes']]

id3 = ID3()
tree = id3.buildtree(data)
print(id3.classify(['clouds',  'cold', 'normal', 'strong'], tree))
# output: {'decision': 'yes'}

```

#### Unique Counts:

```python
from thinkmachine import ID3
# Frequency of values of a given column.
# By default the frequency is the of last column.
#
# Col:    weather   temperature   humidity   wind       decision    
# index:  0         1             2          3          4
data = [['rain',    'cold',       'high',   'weak',     'yes'],
        ['sunny',   'hot',        'high',   'weak',     'no'],
        ['clouds',  'pleasant',   'high',   'weak',     'yes'],
        ['clouds',  'pleasant',   'normal', 'strong',   'no'],
        ['sunny',   'cold',       'high',   'weak',     'yes'],
        ['sunny',   'pleasant',   'high',   'weak',     'no'],
        ['clouds',  'cold',       'normal', 'strong',   'yes']]

id3 = ID3()

print(id3.uniquecounts(data, 0))
# output: {'rain': 1, 'sunny': 3, 'clouds': 3}
```

#### Entropy:

```python
from thinkmachine import ID3

data = [['rain',    'cold',       'high',   'weak',     'yes'],
        ['sunny',   'hot',        'high',   'weak',     'no'],
        ['clouds',  'pleasant',   'high',   'weak',     'yes'],
        ['clouds',  'pleasant',   'normal', 'strong',   'no'],
        ['sunny',   'cold',       'high',   'weak',     'yes'],
        ['sunny',   'pleasant',   'high',   'weak',     'no'],
        ['clouds',  'cold',       'normal', 'strong',   'yes']]

id3 = ID3()
print(id3.entropy(data))
# output: 0.9852281360342516
```

### LICENSE
This project is licensed under the MIT License. This means you can use and modify it for free in private or commercial projects.


### Development

  $ git clone https://github.com/Wellington475/ThinkMachine
  $ pip install -r requirements.txt
  $ nosetests

  
