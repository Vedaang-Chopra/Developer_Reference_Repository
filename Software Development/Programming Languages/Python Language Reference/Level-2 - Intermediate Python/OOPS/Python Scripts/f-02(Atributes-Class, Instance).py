#  Class Atrributes: Set of variables holding data.
        #  They always lie within the class, so can only be reffered from class or object. They are variables but only for classes
        #  There are two types, class and Instance atrributes.
        
class Expense_Tracker:
    #  Docstring: - Description of class
    '''
    This is a class to do Expense Tracking

    '''
    #  Class Attribute.......................
    #  Class attribute: They remain unique to the class, and not different for each object. Same value for each object(same memory)
    expense_tracker_version=0.1
    #  This/Self keyword is reference to current instance to the class. 
    def __init__(this,tracker_category,opening_balance,budget) -> None:
        # Instance Atrributes..........
        # Instance attribute: They remain unique to the object, and different for each object. Different value for each object(different memory)
    
        this.tracker_category=tracker_category
        this.opening_balance=opening_balance
        this.budget=budget
        


#  Creating an object, from class
Home_Tracker=Expense_Tracker('home',3000, 5000)
print(Home_Tracker.tracker_category)
Shopping_Tracker_Debit =Expense_Tracker('shopping', 1000,10000)
print(Shopping_Tracker_Debit.tracker_category)

#  Some of the inbuilt functions for classes and objects...............
#  Dict Returns the values of all attributes for that object, It doesn't show the class attributes.
print(Home_Tracker.__dict__)


# getattr: - Return The attribute value, if it exists. If not returns error
print(getattr(Shopping_Tracker_Debit, 'opening_balance'))

#  hasattr: - Returns true or false, to check if attributes, exists or not
print(hasattr(Shopping_Tracker_Debit, 'opening_balance'))
print(hasattr(Shopping_Tracker_Debit, 'amount'))

#  delattr: - Helps to delete an attribute from an object, Returns nothing
print(delattr(Shopping_Tracker_Debit, 'budget'))
print(Shopping_Tracker_Debit.__dict__)



#  Modifying Class Attributes.........
print('Home Tracker Class Attribute: ',Home_Tracker.expense_tracker_version)
Home_Tracker.expense_tracker_version=0.2
print('Home Tracker Class Attribute: ',Home_Tracker.expense_tracker_version)
print('Shopping Tracker Class Attribute: ',Shopping_Tracker_Debit.expense_tracker_version)
# After modifying the class attribute from one instance it doesn't change the value for all the instances, so the declaration have to quite careful.
# To change the value, have to change it in all instances


