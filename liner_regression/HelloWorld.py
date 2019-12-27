import numpy as np
# print(keyword.kwlist)
mat = np.mat(np.random.rand(4,4))
print(mat, mat.I)

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

