# If-Else Condition....................................
# Note:-
#  1. While using if else we can avoid ( condition ) brackets.
#  2. The brackets/block which was in c++/java is replaced by indentation of code. Thus indentation is must.
#  3. Instead of else-if we use elif.
# Syntax:-
a=int(input())
b=int(input())
c=int(input())
if a >= b:
    if a >= c:
        print(a)
    else:
        print(c)
else:
    if b >= c:
        print(b)
    else:
        print(c)

# Loops.......................................................

# Loop 1:- While Loop:
# Note:-
#  1. While using while loop we can avoid ( condition ) brackets.
#  2. The brackets/block which was in c++/java is replaced by indentation of code. Thus indentation is must.
# Syntax:-
print('Using Loops:Printing numbers from 1 to n')
n=int(input())
i=1
while i<=n:
    print(i,end=" ")
    i=i+1
print('\n Numbers printed')

# Loop 2:-For Loop:
# Note-The range function is used to create a specific range from the parameters given.
print(range(1,10))
print(type(range(1,10)))
# The range function can take 3 parameters.