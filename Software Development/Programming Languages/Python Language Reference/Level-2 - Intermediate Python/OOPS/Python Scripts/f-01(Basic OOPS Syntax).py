# Object Oriented Programming: - Divide the code, into managable components

#  Class: - Blueprint or General idea of the program
    # Class Attributes: General variables holding data about the class
    # Methods: Functions that perform, actions on such attributes

# Object: - Instances of Class, Multiple Unique Copies of Class 

class Expense_Tracker:
    #  Docstring: - Description of class
    '''
    This is a class to do Expense Tracking

    ''' 
    #  Constructor of Class.........
    #  Purpose is to assign value to attributes, from outside of the class
    def __init__(self,date,description,amount) -> None:
        #  Self, is like this keyword. Used to refer the internal attributes of the class.
        #  Here instead of the word "self", we can have any other word, just need to define it in constrcutor
        self.date=date
        self.description=description
        self.amount=amount


#  Creating an object, from class
Home_Tracker=Expense_Tracker('4th June 2022','Snacks Expense', 1500)
print(Home_Tracker.description)
Office_Tracker_Debit =Expense_Tracker('2nd June','Flight Tickets', 10000)
Office_Tracker_Credit =Expense_Tracker('25th June','Salary Credited', 50000)

print(Office_Tracker_Debit.amount,Office_Tracker_Credit.amount)

