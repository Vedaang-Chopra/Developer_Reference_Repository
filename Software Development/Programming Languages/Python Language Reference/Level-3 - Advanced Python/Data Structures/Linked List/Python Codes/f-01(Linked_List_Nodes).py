#  Linked list is a data structure, that stores data sequentially but not in a continuous manner in memory.
#  Every element in linked list is called node.

class Linked_List_Node:
    head=None
    
    def __init__(self,data,next) -> None:
        self.data=data
        self.next=next

#  Reference to First Node, is called Head and needs to be kept in memory.
#  Reference to Last Node, is called Tail.

a=Linked_List_Node(13,None)
b=Linked_List_Node(15,None)
a.next=b

print(a.data)
print(b.data,a.next.data)
