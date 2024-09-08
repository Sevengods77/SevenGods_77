def generate_star_pattern(N):
    for i in range(N):
        stars = N - i
        for j in range(stars):
            if j % 5 == 4:
                print('#', end='')
            else:
                print('*', end='')
        print()
T = int(input())
for _ in range(T):
    N = int(input())
    generate_star_pattern(N)