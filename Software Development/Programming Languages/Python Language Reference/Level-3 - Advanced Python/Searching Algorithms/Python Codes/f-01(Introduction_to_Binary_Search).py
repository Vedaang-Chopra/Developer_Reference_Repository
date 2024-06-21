




def binary_search(arr:list, element, debug=False):
    lower=0
    upper=len(arr)-1
    while lower <= upper:
        mid=int((lower+upper)/2)      # mid = (lower+upper)//2 Integer Division
        if debug:
            print("Lower, Upper Bounds and Middle are:", lower, upper, mid)
            print("Array Values for Lower, Upper Bounds and Middle are:", arr[lower], arr[upper], arr[mid])
        if element == arr[mid]:
            return "Element Found at index, "+str(mid), mid
        elif element > arr[mid]:
            lower = mid+1
        else:
            upper = mid - 1
    
    return "Element Not Found in the array", -1


arr=[12,34,45,56,56,56,56,67,89]
x1=56
x2=57
print(binary_search(arr, x1, False))
print(binary_search(arr, x2, False))