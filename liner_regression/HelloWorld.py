import numpy as np
# print(keyword.kwlist)

a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2,3,4,5], dtype=complex)
print(a, b, a.ndim, a.shape, a.size, a.dtype, a.itemsize, b.flags)

print(np.c_[np.array([[1,2,3]]),0, 0, np.array([[4,5,6]])])

a = np.arange(24)
print(a, a.ndim)
a = a.reshape(2,4,3)
print(a, a.ndim)
print(np.eye(4))

x = np.zeros((4, 1))
print(x)



'''
input = "I like programming"
inputWords = input.split(" ")
inputWords = inputWords[-1::-1]
output = ' '.join(inputWords)
print(output)

student = {'Tom', 'Jim', 'Mary', 'Tom', 'Jack', 'Rose'}
print(student)
if 'Rose' in student:
    print("Rose is in the set")
else:
    print("Rose is not in the set")

a = ['a', 'b', 'c']
n = [1,2]
b = [a, n]
print(b[1])


class A:
    pass
class B(A):
    pass
print(isinstance(A(), A), isinstance(B(), A), type(A()) == A, type(B()) == A)

a, b, c, d = 20, 5.5, True, 4 + 3j
print(type(a), type(b), type(c), type(d), isinstance(a, int))

str = "Liufeng"
print(str)
print(str[0:-1])
print(str[0])
print(str[2:5])
print(str[2:])
print(str * 5)
print("Hello, World")
t = input("\n\n按下enter键退出")
print(t)
'''

