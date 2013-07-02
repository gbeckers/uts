def h5attr_to_geometry(attr, pointnames=None):
    from uts.core.geometry import RectangularLattice
    code = '%s(' % attr[0] #class
    for key,item in attr[1].items(): #parameters
        code += '%s=%s,' % (key,item)
    if not pointnames is None:
        code += '%s=%s,' % ('pointnames', list(pointnames))
    code =code[:-1] + ')'
    return eval(code)