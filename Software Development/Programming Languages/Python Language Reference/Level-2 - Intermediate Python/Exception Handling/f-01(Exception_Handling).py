#  Errors vs Exceptions........
# https://www.geeksforgeeks.org/errors-v-s-exceptions-in-java/

# Error :- Flaw in Code (Compile time, Run time, Logical Error)
# Exception: - Disrruption in normal working of code


#  Exception Handling
# Try- Catch(Except) Blog



try:
    a=1
    b='s'
    c=a+b
except NameError as err:
    print("Undefined Variable")
    print(err.args)
except Exception as err:
    print("Some problem may have occurred")
    # print(err.args)
    # print(err.with_traceback())
    print(err)