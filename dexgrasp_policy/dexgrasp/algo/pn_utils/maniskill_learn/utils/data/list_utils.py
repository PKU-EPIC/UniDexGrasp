def auto_pad_lists(a, b):
    """
    Input two objects, then output two list of objects with the same size.
    """
    if not isinstance(a, (list, tuple)):
        a = [a]
    if not isinstance(b, (list, tuple)):
        b = [b]
    if len(a) > len(b):
        for i in range(len(a) - len(b)):
            b.append(a[0])
    elif len(a) < len(b):
        for i in range(len(b) - len(a)):
            a.append(b[0])
    return a, b