def nested(t):
    for i in t:
        if isinstance(i, tuple):
            yield from nested(i)
        else:
            yield i

# tup = ((1, 2, 3), (4, 5, (6, 7)), 8, 9)
# for elem in nested(tup):
#     print(elem)