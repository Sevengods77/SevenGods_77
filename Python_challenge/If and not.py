Test_case=int(input())
for i in range (0,Test_case):
    n=int(input())
    for j in range (0,n):
        if j==0:
            print((j+3),end=" ")
        elif j%2==0:
            print(j*2,end=" ")
        elif j%2!=0:
            print(j**2,end=" ")
    print()



