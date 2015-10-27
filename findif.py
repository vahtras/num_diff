import numpy
from attributes import *
DELTA = 1e-5

def grad(f, delta=DELTA):
    def grad_f(*args, **kwargs):
        if len(args) == 1:
            x, = args
            gradf_x = (
                f(x+delta/2) - f(x-delta/2)
                )/delta
            return gradf_x
        elif len(args) == 2:
            x, y = args
            if type(x) in [float, int] and type(y) in [float, int]:
                gradf_x = (f(x + delta/2, y) - f(x - delta/2, y))/delta
                gradf_y = (f(x, y + delta/2) - f(x, y - delta/2))/delta
                return gradf_x, gradf_y
    return grad_f

def hessian(f, delta=DELTA):
    def hessian_f(*args, **kwargs):
        if len(args) == 1:
            x, = args
            hessianf_x = (
                f(x+delta) + f(x-delta) - 2*f(x)
                )/delta**2
            return hessianf_x
        elif len(args) == 2:
            x, y = args
            if type(x) in [float, int] and type(y) in [float, int]:
                hess_xx = (f(x + delta, y) + f(x - delta, y) - 2*f(x, y))/delta**2
                hess_yy = (f(x, y + delta) + f(x, y - delta) - 2*f(x, y))/delta**2
                hess_xy = (
                    f(x+delta/2, y+delta/2) 
                  + f(x-delta/2, y-delta/2) 
                  - f(x+delta/2, y-delta/2) 
                  - f(x-delta/2, y+delta/2)
                    )/delta**2
                return hess_xx, hess_xy, hess_yy
    return hessian_f

def ndgrad(f, delta=DELTA):
    def grad_f(*args, **kwargs):
        x, = args
        grad_val = numpy.zeros(x.shape)
        it = numpy.nditer(x, op_flags=['readwrite'], flags=['multi_index'])
        for xi in it:
            i = it.multi_index
            xi += delta/2
            fp = f(x)
            xi -= delta
            fm = f(x)
            xi += delta/2
            grad_val[i] = (fp - fm)/delta
        return grad_val
    return grad_f

def ndhess(f, delta=DELTA):
    def hess_f(*args, **kwargs):
        x, = args
        hess_val = numpy.zeros(x.shape + x.shape)
        it = numpy.nditer(x, op_flags=['readwrite'], flags=['multi_index'])
        for xi in it:
            i = it.multi_index
            jt = numpy.nditer(x, op_flags=['readwrite'], flags=['multi_index'])
            for xj in jt:
                j = jt.multi_index
                xi += delta/2
                xj += delta/2
                fpp = f(x)
                xj -= delta
                fpm = f(x)
                xi -= delta
                fmm = f(x)
                xj += delta
                fmp = f(x)
                xi += delta/2
                xj -= delta/2
                hess_val[i + j] = (fpp + fmm - fpm - fmp)/delta**2
                print hess_val
        return hess_val
    return hess_f

def clgrad(obj, exe, arg, delta=DELTA):
    f, x = get_method_and_copy_of_attribute(obj, exe, arg)
    def grad_f(*args, **kwargs):
        grad_val = numpy.zeros(x.shape)
        it = numpy.nditer(x, op_flags=['readwrite'], flags=['multi_index'])
        for xi in it:
            i = it.multi_index
            xi += delta/2
            fp = f(*args, **kwargs)
            xi -= delta
            fm = f(*args, **kwargs)
            xi += delta/2
            grad_val[i] = (fp - fm)/delta
        return grad_val
    return grad_f
        


            
