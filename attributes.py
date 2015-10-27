def get_method_and_copy_of_attribute(obj, exe, arg):
    setattr(obj, arg, getattr(obj, arg).copy())
    f = getattr(obj, exe)
    x = getattr(obj, arg)
    return f, x
