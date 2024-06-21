#  Iterables:
#  List is an example of iterable. We can iterate through the values of a list, since all values have a refernce present in the memory. 
# E.g: - 
lst=[0,1,2,3,4,5,6,7,8,9]
for i in lst:
    print(i, end="     ")
print()

#  Iterators: 
# An iterator is an example of data structure, where not all values are present in the memory.
# It is only available in the memory, once we call next function and only a single value can be present in the memory.
# TO avoid consuming large memory, we create iterators.

# Creating an iterator.....
#  Iter Function : - Creates an iterable to an iterator.
iterator_list=iter(lst)
print("Created Iterator:",iterator_list)

# To access values, we use next function. Next acts a pointer to values, and the iterator behaves similar to a singly linked list. 
# With every next, we can refer the next value.
# To access the first value, again, create the iterator again, to reset next
try:
    
    print("First value of iterator:", next(iterator_list))
    print("Second value of iterator:", next(iterator_list))
except StopIteration:   # Use this to check whteher next hasn't exceeded the last element of iterator
    print("Empty Iterator, nill refernce of next") 

# After the last element, if done next there is error thrown (Stop iteration).

# Reseting the iterator....
iterator_list=iter(lst)


# For loop has inbuilt exception handling for last element.
print("To print values/ access values, using for loop function....")
for i in iterator_list:
    print(i, end="      ")
