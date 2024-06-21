import math as m
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print(type(x))
x.dtype
print(m.sqrt(12))
z = [2, 3, 4]
z=np.array(z,np.int32)
print(type(z))
print(z)
print(z.dtype)
a=np.zeros((2,3))
print(a)
a1=np.array([[1,2],[3,4]])
a2=np.array([[10,20],[30,40]])
print(a1,a2,a1.dot(a2),end=" ")