#  Methods in Python (OOPS):
    # Class Method: -
    # Static Method: -
    # Instnace Method: -

#  Static Methods: - 



class Expense_Tracker:
    expense_tracker_version=0.1
    def __init__(self,tracker_category,opening_balance,budget) -> None:
        self.tracker_category=tracker_category
        self.opening_balance=opening_balance
        self.budget=budget

    #  Instance Method
    #  Self has to be a parameter, always for instance method
    def get_opening_balance(self):
        return self.opopening_balance


#  Creating an object, from class 
Home_Tracker=Expense_Tracker('home',3000, 5000)
print(Home_Tracker.tracker_category)
Shopping_Tracker_Debit =Expense_Tracker('shopping', 1000,10000)
print(Shopping_Tracker_Debit.tracker_category)