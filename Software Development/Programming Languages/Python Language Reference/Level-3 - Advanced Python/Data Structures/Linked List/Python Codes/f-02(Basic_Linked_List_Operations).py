class Linked_List_Node:
    def __init__(self,data,next) -> None:
        self.data=data
        self.Next=next        

class Linked_List(Linked_List_Node):
    def __init__(self) -> None:
        self.Head=None
        # super().__init__(data, next)
    
    # The function, sets the first node by setting the head pointer
    def set_head(self,first_node):
        self.Head=first_node
    
    #  Attaches a new node to the last node
    def set_next_node(self,current_node,new_node):
        current_node.Next=new_node
        return new_node
    
    def identify_last_node_in_linked_list(self):
        current_reference_to_tail_of_linked_list=self.Head
        while current_reference_to_tail_of_linked_list.Next!=None:
            current_reference_to_tail_of_linked_list=current_reference_to_tail_of_linked_list.Next
        return current_reference_to_tail_of_linked_list
    
    # Optimized Linked List Insertion Code, no need to identify last node again and again........
    # Here for inserting N elements, we have Time Complexity of O(n), since for each node we have 0(1) complexity 
    def linked_list_creation(self,input_data):                    
        # The point of the insertion function is to populate the head, and return or set the value.... 
        # Iterating over the array which holds data for the Linked List
        print("Creating a Linked List and Inserting Data................")
        reference_to_tail_of_linked_list=None
        for i in range(0,len(input_data)):
            CURRENT_DATA=input_data[i]
            print("Recieved Current Data:", CURRENT_DATA)
            #  If input recieved is -1, then inserion of linked list stops
            if CURRENT_DATA==-1:
                print("Recieved -1, hence breaking Linked List Insertion")
                break
            else:
                # Creating a node, with the data
                new_node=Linked_List_Node(CURRENT_DATA,None)
                print("Created Linked list node with data: ", new_node, new_node.data)
                if self.Head==None:             # Check whether the list is empty (are we creating the first node)
                    print("Empty Linked List, Hence populating Head Pointer Value......")
                    self.set_head(new_node)     # Assign Head, to first 
                    reference_to_tail_of_linked_list=self.Head
                    print("After Insertion, Current Linked List Pointer Points to: ", reference_to_tail_of_linked_list, reference_to_tail_of_linked_list.data)
                else:
                    # Attaching other nodes, to previous node
                    if reference_to_tail_of_linked_list==None:
                        raise Exception("Error in Insertion logic, Please check. Linked List iterator wasn't assigned Head value")
                    else:
                        if reference_to_tail_of_linked_list.Next==None:
                            reference_to_tail_of_linked_list=self.set_next_node(reference_to_tail_of_linked_list,new_node)
                            print("After Insertion, Current Linked List Pointer Points to: ", reference_to_tail_of_linked_list, reference_to_tail_of_linked_list.data)
                        else:
                            print("Found Some Error in Linked List Insertion, Hence Identifying the Last node, and inserting new noe after it.")
                            reference_to_tail_of_linked_list=self.identify_last_node_in_linked_list()
                            print("Identified Last node of Linked List is: ", reference_to_tail_of_linked_list, reference_to_tail_of_linked_list.data)
                            reference_to_tail_of_linked_list=self.set_next_node(reference_to_tail_of_linked_list,new_node)
                            print("After Insertion, Current Linked List Pointer Points to: ", reference_to_tail_of_linked_list, reference_to_tail_of_linked_list.data)
            print()
    
    #  Function that prints all values of Linked List
    def print_all_values_of_linked_list(self):
        current_list_pointer=self.Head
        while current_list_pointer is not None:
            print(current_list_pointer.data, end="->")
            current_list_pointer=current_list_pointer.Next
        print("None")
        
        
        
        
def space_sperated_string_input(input_String:str):
    input_data=[int(i) for i in input_String.split(' ')]
    return input_data



# input_string=input("Please enter space seperated numbers for input to linked list: ").strip()
# input_data=space_sperated_string_input(input_string)

# Setting Default Value for testing
input_data=[1,2,3,4,5,6,7,8,9,-1]
print("Received Data:", input_data)

linked_list_obj=Linked_List()
linked_list_obj.linked_list_creation(input_data)

print("Current Linked list object(Inital Status): ", end='')
print(linked_list_obj)

print("Final Status of data within the Linked List: ",end="")
linked_list_obj.print_all_values_of_linked_list()