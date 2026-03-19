lis1=[]
for i in range(0,6):
    a=float(input())
    lis1.append(a)
dict1=tuple(lis1)
for key in dict1:
    s=dict1[0]
    if key <s:
        key = s
print(key)