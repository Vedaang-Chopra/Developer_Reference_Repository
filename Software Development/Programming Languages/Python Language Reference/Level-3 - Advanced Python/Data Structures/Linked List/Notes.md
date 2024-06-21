####  Linked List 

A Linked List is similar to a list/array.
An array has indexes defined for all elements, but a linked list is not a defined indexing. 

A linked list is where the nodes are connected to each other sequentially but there is no indexing defined. 


Node/Linked List Node: - Each element of Linked list
It stores 2 values; One is the Data and Reference to the next Node

All the nodes have a different memory reference, and are not continuous. 



For example if there is a linked list of 5 nodes, then : - 
Node -1: - Data=1; Memory Reference = 0x200; Reference to next node=0x100
Node -2: - Data=2; Memory Reference = 0x100; Reference to next node=0x500
Node -3: - Data=5; Memory Reference = 0x500; Reference to next node=0x800
Node -4: - Data=4; Memory Reference = 0x800; Reference to next node=0x700
Node -5: - Data=3; Memory Reference = 0x700; Reference to next node=Null

Head of the Linked List
Pointer to the first Node is the Head of the Linked List

Having the head of the linked list; allows to traverse the complete linked list;

Tail of the Linked List
Pointer to the last Node is the Tail of the Linked List

Not used that much in context of Linked List


#### Time Complexity of Insertion 

1. For Iterative solution : -
   1. Step-1: - Find Length of Linked List - O(n)
   2. Step-2:- Finding the appropriate postiion - O(i)
   3. Step-3: - Inserting the node: - O(1)

Therefore Complete Time COmplexity is of order of N/O(n)

2. For Recursive Solution: - 
   1. Step-1: - Since we are receursive calling next node to find postion : - O(i)
   2. Step-2:-  Creating New Node: - O(1)

Therefore complixity is of order of I/ O(i) [less than O(n), just till a particular length of array]