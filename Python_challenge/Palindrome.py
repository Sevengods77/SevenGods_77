def Palindrome(s):
    s=s.lower()
    return s==s[::-1]
Test_case=int(input())
for i in range (0,Test_case):
    s=input()
    ans=Palindrome(s)
    if ans:
        print("It is a palindrome")
    else:
        print("It is not a palindrome")