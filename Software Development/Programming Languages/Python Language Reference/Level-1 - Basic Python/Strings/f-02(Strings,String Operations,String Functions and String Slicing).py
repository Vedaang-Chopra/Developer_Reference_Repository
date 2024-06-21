# Strings .......................................................
string0 = "This is a normal string."
string1 = 'Same as previous string.'
string2 = " It's boring. "                      # Using different quotation marks for a single string
string3 = 'She said "He is Boring".'
string4 = "\"This is just boring, He said. "
# Escaping the meaning of a symbol(The programming language meaning of the symbol is subjugated and
# the character is just a normal symbol).
string5 = r"C:\ProgramFiles\new folder"
print(string5)
# r-here signifies that the string to be used is raw string. That is special characters/escape sequence
# used are part of string and do not have their usual programming language meaning. They are just normal symbols.

# Note:-
# 1. Strings are immutable.
# 2. String plus integer doesn't work.
# 3. There is no character in python. Everything is string.
# 4. Indexing of string goes from 0 to len(string)-1


# String Functions.........................................................................
print(len(string1))      # The len function is used to find the length of the string
print(type(string0))     # The type function is used to find the type of the variable
print(string3.upper())   # This returns a string in upper case.
# Note:-
# 1.<string name>.function name() this syntax returns a new string and doesn't change the existing string.
#   The new String has to be stored in some variable
print(string3.lower())   # This returns a string in lower case.
print(string2.strip())   # This is used to remove the leading and trailing spaces in a string.
print(string4.isalpha()) # This is used to check whether a string is completely alphanumeric or not.
print(string1.isdigit()) # This is used to check whether a string is completely numeric or not.


# String Operations.......................................................................
print(string0 + string2)                        # String concat= + symbol
print(string3*2)                                # Creating string copies within itself
print(string0+"is normal string")


# Slicing Strings............................................................
string ='abcdef'
# Indexing- It is a concept where a string characters can be accessed as the string starts from 0 index and goes till length of the string -1.
#           The string characters can also be accessed using negative numbers as the string goes from -ve of length of string till -1 index.

print(string[0])
print(string[-len(string)])
print(string[0] == string[-len(string)])

# Slicing- It is a concept which is used to extract a substring by using indexing.
# Note- 1. While Slicing strings the index that you enter in the square brackets <string name>[x:y] the substring displayed is in through the following conditions:-
#       a] When positive indexing is used:-
#            <string name>[ x : y ] is equivalent to <string name>[ x : y-1](this is actually displayed)
#       b] When negative indexing is used:-
#            <string name>[ -x : -y ] is equivalent to <string name>[ -x : -y+1](this is actually displayed)
#        2. If no number is mentioned in index then the complete string is displayed
#           eg:- <string name>[x:] is equivalent to string starting from index x and ending at the last index
print(string[0:3])
print(string[-4:-1])
print(string[3:])
print(string[:-3])
print(string[:])
