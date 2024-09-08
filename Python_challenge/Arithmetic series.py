from functools import reduce
def generate_AP(a1,d,n):
    AP_series=[a1]
    for i in range(1,n):
        a1=a1+d
        AP_series.append(a1)
    return AP_series

Test_case=int(input())
for i in range (0,Test_case):
    numbers=input()
    numbers_list=numbers.split()
    a1=int(numbers_list[0])
    d = int(numbers_list[1])
    n = int(numbers_list[2])
    AP_p=generate_AP(a1,d,n)
    for i in AP_p:
        print(i,end=' ')
    print("\n")
    sqr_AP_series=list(map(lambda a:a**2,AP_p))
    for i in sqr_AP_series:
        print(i,end=' ')
    print("\n")
    sum_sqr_AP_series = reduce(lambda a,b:a+b,sqr_AP_series)
    print(sum_sqr_AP_series)

