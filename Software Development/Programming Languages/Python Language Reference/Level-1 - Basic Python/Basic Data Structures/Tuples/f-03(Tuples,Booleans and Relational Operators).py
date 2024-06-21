# Tuples- It is a data structure used to hold multiple data together
a=(1,'Vedaang',3.4)       # Creating a tuple.
print(type(a))
# Accessing tuple elements
print(a[0])
# Note:-
# 1. Tuple is immutable. That is parts of tuple can  not be changed
# 2. It is non homogeneous in nature.
# 3. Assignment doesn't work on tuple items. They can not be changed.(Immutable)
# 4. Slicing works on it exactly like strings.
# 5.
# Operations on tuple
t=(1,4,'3232',334.222)
print(t+a)          # Concatenation of two tuples
print(t*2)          # Creating multiple copies of same data items of same tuple
print(1 in a)       # Checking whether data item exist in tuple
print(3 in t)


# Booleans..........................................
d=True
e=False
print('d=',d,type(d),'\n','e=',e,type(e))

# Relational Operators.............................
x,y=10,20
print(x>y)          # Greater than operator
print(x>=y)         # Greater than equal to than operator
print(x<y)          # Less than operator
print(x<=y)         # Less than equal to than operator
print(x==y)         # Equal to than operator
print(x!=y)         # Not equal to than operator

print(x >= y and x<=y)      # and operator is same as && operator
print(x<=y or x>=y)         # or operator is same as || operator
