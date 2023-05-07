
def split_num(num, n):
    """
    Divide num into m=min(n, num) elements x_1, ...., x_n, where x_1, ..., x_n >= 1 and max_{i,j} |x_i - x_j| <= 1
    """
    n = min(num, n)
    min_steps = num // n
    splits = []
    for i in range(n):
        if i < num - min_steps * n:
            splits.append(min_steps + 1)
        else:
            splits.append(min_steps)
    assert sum(splits) == num
    return n, splits
