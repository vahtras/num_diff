def get_method_and_copy_of_attribute(obj, meth, attr):
    global x_orig
    attr_path = attr.split('.')
    meth_path = meth.split('.')

    f = reduce(getattr, meth_path, obj)
    x_orig = reduce(getattr, attr_path, obj)
    x = x_orig.copy()

    xhead = reduce(getattr, attr_path[:-1], obj)
    xtail = attr_path[-1]
    setattr(xhead, xtail, x)

    return f, x

def reset_attribute(obj, attr):
    global x_orig
    attr_path = attr.split('.')
    xhead = reduce(getattr, attr_path[:-1], obj)
    xtail = attr_path[-1]
    setattr(xhead, xtail, x_orig)
