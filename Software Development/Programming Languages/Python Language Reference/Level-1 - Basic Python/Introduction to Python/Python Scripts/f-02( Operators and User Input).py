# Operators........................................................
print(a+b)                      # Normal Addition
print(a-b)                      # Normal Subtraction
print(a*b)                      # Normal Multiplication
print(a/b)                      # Division with float answer
print(a//b)                     # Division with integer answer
print(a**b)                     # Exponent operator
print(a%b)                      # Remainder Operator
print(a*b/+a/b/b*b**b*b//a-b)   # This is done by BODMAS Rule


# User Input........................................
a=input()
print(a,type(a))
# This input function is used for console input. It returns console input as string. Thus type a would be string.
b=int(input())
print(b,type(b))
# To convert a string into a number we can do it using int function. Similarly there is a float function.
c=float(input())
d=complex(c,b)
print(d,type(d))