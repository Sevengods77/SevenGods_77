def dec_to_binary(n):
    for i in range(7,-1,-1):
        k=n>>i
        if (k&1):
            print("1",end="")
        else:
            print("0",end="")
Test_case=int(input())
for i in range (0,Test_case):
    dec=int(input())
    dec_to_binary(dec)
    print()

