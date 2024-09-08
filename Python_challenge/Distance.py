import math as m
def compute_distance(x1,y1,x2,y2):
    cartesian_distance=m.sqrt((x2-x1)**2+(y2-y1)**2)
    return cartesian_distance

Test_case=int(input())
for i in range (0,Test_case):
    numbers = input()
    numbers_list = numbers.split()
    x1 = int(numbers_list[0])
    y1= int(numbers_list[1])
    x2 = int(numbers_list[2])
    y2 = int(numbers_list[3])
    print("Distance:",format(compute_distance(x1,y1,x2,y2),".2f"))
