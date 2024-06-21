'''
    Python Assert Keyword: -
    The assert statement is a form of exception handling.
    It is a form of if-else statement, but here if the condition or logical expression is false, it throws an AssertionError kind of exception handling.
    It is used where we want to break the code, when a if case fails.
    An assert error is similar to any other error and can be handled with exception handling
'''
#  Case -1:  Simple Assert Logic

# num=10
# assert num>10


# Case-2: Handling Assert in Try Catch

try:
    num=11
    print("Checking the assert Statement for number:", num)
    assert num > 10
    print("Assert Statement check passed as the number was >10, since our expression is num>10")
except AssertionError:
    print("Assert Statement was false as the number was <=10, since our expression is num>10")