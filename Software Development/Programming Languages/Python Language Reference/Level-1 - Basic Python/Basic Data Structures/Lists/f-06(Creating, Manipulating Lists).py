# Lists ...................
# 1. Lists are actually arrays of C++ and Java but with a slight modification.
# 2. It can store data in continuous memory location but the data can be non-homogeneous.
# 3. It is different from tuples as it is mutable as compared to tuples which are non-mutable.

# Creating Lists............
# Method 1:- Square bracket Method.
a1=[1,2,3,4]
print(type(a1),a1)
# Method 2:- The List Function.
a2=list()                           # Here a2 is an empty list.
print(type(a2),a2)
# Method 3:- Using list function and another list.
a3=list(a1)
print(type(a3),a3)
a3=list([1,2,3,4])
print(type(a3),a3)
a3=list("Hello")                    # a3 is a list of all characters of the word.
print(type(a3),a3)
# Method 4:- Using Range Function.
a4=[0 for i in range(10)]           # a4 here is a list with 10 zeroes.
print(type(a4),a4)
# This syntax is used to create lists using the loop syntax and range function. The "[]" ensures a list.
# The expression before "for"(keyword) is the value added in each iteration.
# The for statement and range function ensures the loop.
a5=[i for i in range(10)]           # a5 here is a list of first 10 numbers.
print(type(a5),a5)
a6=[i*i for i in range(10)]         # a6 here is a list with squares of first 10 numbers.
print(type(a6),a6)

# Non-Homogeneous Lists....
a7=[1,2.4,"as",'c',3+4j]
print(type(a7),a7)

# Accessing Data Elements of List..
# 1. Indexing of lists is same as that of strings.
# 2. Lists allow both positive and negative indexing(to obtain single element).
print("a7 list, 0 element:",a7[0])
print("a7 list, 1 element:",a7[1])
print("a7 list, 3 element:",a7[3])
print("a7 list, -1 element:",a7[-1])
print("a7 list, -3 element:",a7[-3])
# 3. Lists also allow slicing(to obtain multiple elements).
print("a7 list, 0-3 element",a7[0:4])
print("a7 list, -4-(-1) element:",a7[-4:-1])
# Length function:
print("Length function:",len(a7))
# Fast Iterations......................
print("Fast Iteration of A7 List:")
for i in a7:
    print(i)

# Accepting an Input which is space separated.........
str=input()                         # Here str is string
str=str.strip()                     # This is used to remove the leading and trailing spaces.
str=str.split(" ")
# Here str changes into a list with each element as the space separated element of the string.
# All the elements of the list str are strings.
str=[int(i) for i in str]
# This statement changes every element of the string list into an integer list.
print(str,type(str))

# One-Line code for the Accepting an Input which is space separated.........
# str=[int(i) for i in input().strip().split(" ")]

# Adding elements to a List......................
a8=[1,"dsad",3.4,'c',131]
a8.append(3)                        # The append function adds the element to the end of the list.
a8.insert(1,"SDDDDD")
# The insert function adds data(second parameter) at a specific index(first parameter) \
# and shifts all other elements to the right.
a8.extend(a7)                       # The extend function appends the parameter(which is a list) to the current list selected.
print(a8)


# Deleting Elements from the list.............
# Method 1:- Deleting by index:
a8.pop()
# If the pop function has no argument then it deletes the last element of the list.
a8.pop(3)
# Here the parameter is the index of the data which is to be deleted.
del a8[5]
del  a8[0:2]                        # Slicing and deleting
# Here the element at index 5 is deleted.
# Method 1:- Deleting by Data Element:
a8.remove("dsad")
# The remove function removes the first occurence of the element which is passed as the parameter.

# List operations............................
a9=[1,2,3,4,"fsfsf",3.2,3.5,6.7]
# a9+2              # This is not allowed(list+integer)
# Creating Multiple copies:
print(a9*3)
print(a9+a9)
print(a9.extend(a9))


# List Functions
a10=[1,2,3,5,6,3,13,311,441,312,1223,31231,13131]
print(a10.sort())
# Sorting The List with sort function
print(a10.count())
# Gives the total number of elemnts in list
print(a10.count(2))
# Gives the count of the parameter passed within the list.
print(a10.reverse())
# Returns the reversed list.
print(1 in a10)
# The above expression returns a boolean, performing a check whether the data exists in the list or not.

# Sorting in Lists.........................
# Note:-The Bubble and Selection Sort Code is in the Python_code_Practised Directory(filename:-"f-05")
# .



