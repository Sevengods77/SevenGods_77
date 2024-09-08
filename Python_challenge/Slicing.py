Test_case=int(input())
for i in range (0,Test_case):
    list_length = int(input())
    input_list=input().split()
    number_list=[int(x) for x in input_list]
    reverse_list=(number_list[::-1])
    for j in reverse_list:
        print(j,end=" ")
    print()
    add_list=(number_list[3:list_length:3])
    for k in add_list:
        print(k+3,end=" ")
    print()
    subtract_list=(number_list[5:list_length:5])
    for l in subtract_list:
        print(l-7,end=" ")
    print()
    sum_list=(number_list[3:8])
    sum=0
    for m in sum_list:
        sum+=m
    print(sum)





