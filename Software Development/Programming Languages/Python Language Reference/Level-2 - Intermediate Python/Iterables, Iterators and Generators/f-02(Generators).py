# Generators..............
# They are computation technique, which doesn't hold everything in memory/ It yields(returns) one result at a time.
# It performs computation one result at a time. Have to fetch result one at a time.
# Generator vs Iterator: Generator in turns creates an iterator (specific case of Iterator)
    # 1. Iterator is created using iter keyword, and generator is created using yield keyword and a function
    # 2. Generator saves the local variable value, in the function that is used during creation, while iterator doesn't.
    # 3. Generator helps write fast and compact code, Iterator doesn't
    # 4. Iterator much more memory efficient.


#  Advantages: 
#  1. Code much more readable.
#  2. Memory advantage, since not all data is in memory

a=[1,2,3,4,5,6]

#  General Funcution to return square numbers..............
def square_numbers(nums):
    square=[]
    for i in nums:
        square.append(i*i)
    return square


squared_array=square_numbers(a)
print(squared_array)


#  Use of yield keyword, creates a generator
#  Yield keyword: - Used to create generator function
#  The generator is used in case of scale data. It is used to return data without using additional list.
# Example: -
def square_numbers_generator(nums:list):
    for i in nums:
        yield (i*i)

#  Result is a generator object.
generator_result=square_numbers_generator(a)
# To get the result from the generator object, we use the next keyword. It gives result one at time, how much time we run next
print(next(generator_result))
print(next(generator_result))

#  Iterator v/s Generator
print("Check if Iterator and Generator are same:")
import types, collections
print(issubclass(types.GeneratorType, collections.Iterator))

#  X-RANGE v/s RANGE

# from past.builtins import xrange
print("............................XRANGE v/s RANGE.................")
print(xrange(3), type(xrange(3)))
print(range(3), type(range))
# x=range(3)
# print(x[0])
#  Xrange v/s Range: - Returns a generator, where as range returns a list