# Lambda Function: - Anonyous function, function with no name
# If a function has a single expression, then you can convert it to a lmbda function, and store it in a variable
# Function with single expression, of return value
# To avoid creation of function

def addition(a,b):
    return a + b

addition_lambda=lambda a,b:a+b
print(addition(1,2),addition_lambda(1,2))

def even(num):
    if num%2==0:
        return True
    else:
        return False

even_lambda= lambda num:num%2==0

print(even(10),even_lambda(10))