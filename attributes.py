def get_method_and_copy_of_attribute(obj, meth, attr):
    attr_path = attr.split('.')
    meth_path = meth.split('.')

    f = reduce(getattr, meth_path, obj)
    x = reduce(getattr, attr_path, obj).copy()

    xhead = reduce(getattr, attr_path[:-1], obj)
    xtail = attr_path[-1]
    setattr(xhead, xtail, x) 

    return f, x

