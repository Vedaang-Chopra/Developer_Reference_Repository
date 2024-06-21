#  Practicing Lambda Functions, Map and Filter Commands............


def odd_even(number):
    if number%2==0:
        return "even"
    else:
        return "odd"

lst=[i for i in range(0,10)]
print(list(map(odd_even,lst)))

# print(list(filter(odd_even,lst)))
