import numpy

def grad(f, delta=1e-5):
    def grad_f(*args, **kwargs):
        if len(args) == 1:
            x, = args
            if type(x) in (float, int):
                gradf_x = (
                    f(x+delta/2) - f(x-delta/2)
                    )/delta
                return gradf_x
            else:
                if type(x) == numpy.ndarray:
                    grad_val = numpy.zeros(x.shape)
                    for i in range(len(x)):
                        x[i] += delta/2
                        fp = f(x)
                        x[i] -= delta
                        fm  = f(x)
                        x[i] += delta/2
                        grad_val[i] = (fp - fm)/delta
                    return grad_val
                        

        elif len(args) == 2:
            x, y = args
            if type(x) in [float, int] and type(y) in [float, int]:
                gradf_x = (f(x + delta/2, y) - f(x - delta/2, y))/delta
                gradf_y = (f(x, y + delta/2) - f(x, y - delta/2))/delta
                return gradf_x, gradf_y
        
    return grad_f



            
