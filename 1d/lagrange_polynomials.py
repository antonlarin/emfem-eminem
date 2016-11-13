def basis(deg, *args):
    if (deg < 1 or deg > 2):
        raise ValueError('Unsupported Lagrangian polynomial degree')

    if (len(args) != deg + 1):
        raise ValueError('Incorrect number of points')

    if (deg == 1):
        return _lagrange_1_polynomials(*args)
    else:
        return _lagrange_2_polynomials(*args)


def basis_derivatives(deg, *args):
    if (deg < 1 or deg > 2):
        raise ValueError('Unsupported Lagrangian polynomial degree')

    if (len(args) != deg + 1):
        raise ValueError('Incorrect number of points')

    if (deg == 1):
        return _lagrange_1_polynomials_derivatives(*args)
    else:
        return _lagrange_2_polynomials_derivatives(*args)


def _lagrange_1_polynomials(x0, x1):
    return [
            lambda x : (x - x1) / (x0 - x1),
            lambda x : (x - x0) / (x1 - x0)
    ]

def _lagrange_1_polynomials_derivatives(x0, x1):
    return [
            lambda x: 1 / (x0 - x1),
            lambda x: 1 / (x1 - x0)
    ]

def _lagrange_2_polynomials(x0, x1, x2):
    return [
            lambda x : (x - x1) * (x - x2) / (x0 - x1) / (x0 - x2),
            lambda x : (x - x0) * (x - x2) / (x1 - x0) / (x1 - x2),
            lambda x : (x - x0) * (x - x1) / (x2 - x0) / (x2 - x1)
    ]

def _lagrange_2_polynomials_derivatives(x0, x1, x2):
    return [
            lambda x : ((x - x1) + (x - x2)) / (x0 - x1) / (x0 - x2),
            lambda x : ((x - x0) + (x - x2)) / (x1 - x0) / (x1 - x2),
            lambda x : ((x - x0) + (x - x1)) / (x2 - x0) / (x2 - x1)
    ]

