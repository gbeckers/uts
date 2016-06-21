import numpy as np

from .check_arg import check_int, check_float

## FIXME missing points

class GeometryError(Exception): pass

# class Point(object):

#     def __init__(self, index, coordinate=None, name=None, geometry=None):

#         self.__index = index
#         self.__coordinate = coordinate
#         self.__name = name
#         self.__geometry = geometry

#     index = property(lambda self: (self.__index), None, None, "")
#     coordinate = property(lambda self: (self.__coordinate), None, None, "")
#     name = property(lambda self: (self.__name), None, None, "")
#     geometry = property(lambda self: (self.__geometry), None, None, "")


class Geometry(object):
    """
    Superclass for geometries.

    """

    pass


class RectangularLattice(Geometry):
    """
    *   *   *   *   *   *
    *   *   *   *   *   *
    *   *   *   *   *   *
    *   *   *   *   *   *

    0   1   2   3   4   5
    6   7   8   9   10  11
    12  13  14  15  16  17
    18  19  20  21  22  23
    
    """

    def __init__(self, nx, ny, pointnames, dx=None, dy=None):

        self.__nx = check_int(nx, negative=False, zero=False, cast=int)
        self.__ny = check_int(ny, negative=False, zero=False , cast=int)
        self.__npoints = self.__nx * self.__ny
        if dx is None:
            self.__dx = None
        else:
            self.__dx = check_float(dx, negative=False, zero=False)
        if dy is None:
            self.__dy = None
        else:
            self.__dy = check_float(dy, negative=False, zero=False)
        self.__pointnames = self._check_names(pointnames, self.__npoints)
        self.__pointdict = self._to_dict(self.__pointnames)


    nx = property(lambda self: self.__nx, None, None, "")
    ny = property(lambda self: self.__ny, None, None, "")
    dx = property(lambda self: self.__dx, None, None, "")
    dy = property(lambda self: self.__dy, None, None, "")
    shape = property(lambda self: (self.__nx, self.__ny), None, None, "")
    npoints = property(lambda self: self.__npoints, None, None, "")
    pointnames = property(lambda self: self.__pointnames,
                                 None, None, "")
    
    def info(self):
        s = ''
        pn = ''
        for row in self.pointnames:
            pn += '    '
            for col in row:
                pn += '%s  ' %col
            pn += '\n' 
                
        return 'geometry:\n%s\n' % pn
    
    
    def __eq__(self, other):
        result = True
        result &= (self.nx == other.nx)
        result &= (self.ny == other.ny)
        result &= (self.dx == other.dx)
        result &= (self.dy == other.dy)
        result &= (self.npoints == other.npoints)
        result &= (self.pointnames == other.pointnames).all()
        return result
        

    def __getitem__(self,index):
        try:
            item = str(index)
        except:
            raise TypeError("cannot convert index to string")
        if not item in self.pointnames:
            raise IndexError("no point with name %s exists" % item)
        return self.__pointdict[item]
    
    def __str__(self):
        return '\n'.join(['   '.join(point) for point in self.pointnames])
            
    def __repr__(self):
        return "<RectangularLattice> nx: %d ny: %d" \
               % (self.nx, self.ny)

    # FIXME duplicate pointnames should be renamed?
    def _check_names(self, pointnames, npoints):
        """
        Checks if pointnames parameter is properly formatted. Will return
        a 2-D numpy array (rows x columns) of type string with the
        pointnames.
        """

        #if pointnames is None:
            #s = 'point_%%.%dd' % max(1, int(np.ceil(np.log10(npoints))))
            #pointnames = [s % i for i in range(npoints)]
        try:
            pointnames = np.array(pointnames)
        except:
            raise TypeError("cannot convert pointnames to string array")
        try:
            pointnames = pointnames.flatten()
            pointnames = pointnames.reshape(self.ny, self.nx)
        except:
            raise GeometryError("pointnames (size=%d) not congruent with lattice structure (shape=%dx%d)" \
                                % (pointnames.size, self.ny, self.nx))

        if not np.issubdtype(pointnames.dtype, np.string_):
            raise GeometryError("pointnames name(s) not of string type")
        if not np.unique(pointnames).size == pointnames.size:
            raise GeometryError("pointnames not unique")

        return pointnames

    def _to_dict(self, pointnames):
        '''
        pointnames is a 2D array; this has been checked before
        '''
        pointdict = {}
        for rown, row in enumerate(pointnames):
            for coln, pointname in enumerate(row):
                pointdict[pointname] = (rown,coln)   
        return pointdict
    
    # FIXME check that this works for more dimensions!
    def mapfrom1Dto1D(self, array, pointnames):
        '''
        maps a 1-d array with given pointnames to a 1-d array of geometry
        '''
        if not array.shape[-1] == self.npoints:
            raise GeometryError("shape of array not congruent with number of points")
        if not len(pointnames) == self.npoints:
            raise GeometryError("number of pointnames not congruent with number of points")
        newarray = np.empty_like(array)
        newarray[...,self.indexfrom1Dto1D(pointnames)] = array
        return newarray
        
    def mapfrom1Dto2D(self, array, pointnames, hflip=False):
        '''
        Maps a 2-d array with given pointnames to a 1-d array of geometry.
        The last axis of the array is mapped to two axes representing 
        the recangular lattice of geometry.
        '''
        if not array.shape[-1] == self.npoints:
            raise GeometryError("shape of array not congruent with number of points")
        if not len(pointnames) == self.npoints:
            raise GeometryError("number of pointnames not congruent with number of points")
        newshape = list(array.shape)[:-1]
        newshape.extend([self.ny,self.nx])
        newarray = array.reshape(-1,self.ny,self.nx).copy()
        for i,el in enumerate(newarray):
            el[self.indexfrom1Dto2D(pointnames, hflip=hflip)] = el.flatten()
        return newarray.reshape(newshape)
        
    def mapfrom2Dto1D(self, array, pointnames):
        '''
        maps the last 2 axes of an array with given pointnames to a one axes
        corresponding to the linear layout of geometry, i.e. flattened
        '''
        pointnames = np.array(pointnames)
        if not pointnames.size == self.npoints:
            raise GeometryError("shape of array not congruent with number of points")
        newshape = list(array.shape)[:-2]
        newshape.append(self.ny*self.nx)
        newarray = array.reshape(-1,self.ny*self.nx).copy()
        for i,pointname in enumerate(pointnames):
            el = newarray[...,i]
            el[self.indexfrom2Dto1D(pointnames)] = el
        return newarray.reshape(newshape)

    def indexfrom1Dto1D(self, pointnames):
        if not len(pointnames) == self.npoints:
            raise GeometryError("number of pointnames not congruent with number of points")
        indices = []
        for pointname in pointnames:
            row, col = self.__pointdict[pointname]
            indices.append(row*self.nx+col)
        return np.array(indices)

    def indexfrom1Dto2D(self, pointnames, hflip=False):
        if not len(pointnames) == self.npoints:
            raise GeometryError("number of pointnames not congruent with number of points")
        indices = []
        for pointname in pointnames:
            indices.append(self.__pointdict[pointname])
        indices = np.array(indices).T
        if hflip:
            indices[1] = np.abs(indices[1]-7)
        return indices[0], indices[1]
        
    def indexfrom2Dto1D(self, pointnames):
        if not (pointnames.size == self.npoints):
            raise GeometryError("number of pointnames not congruent with number of points")
        indices_i = []
        indices_j = []
        for pointname in self.pointnames.flatten():
            i,j = np.where(pointnames==pointname)
            indices_i.append(i.item())
            indices_j.append(j.item())
        return self.nx*np.array([indices_i])+np.array([indices_j])

    def instantiationcode(self):
        """Returns a length-2 tuple containing (0) the name of the class, and
        (1) a dictionary of the instantiation parameters.

        Needed to be able to save as a h5 node attribute

        Does not include pointnames
        """
        return ('RectangularLattice', {'dx': self.dx, 'dy': self.dy,
            'nx': self.nx, 'ny': self.ny})
            #,'pointnames': self.pointnames.tolist()})
    
    def distance(self, pointname1, pointname2):
        if (self.dx is None) or (self.dy is None):
            raise ValueError("no information on absolute point positions")
        c1 = np.array(self.__pointdict[pointname1]) * (self.dy, self.dx)
        c2 = np.array(self.__pointdict[pointname2]) * (self.dy, self.dx)
        return np.power(np.power(c1-c2,2.0).sum(),0.5)
    
    def distance_matrix(self, pointnames):
        if (self.dx is None) or (self.dy is None):
            raise ValueError("no information on absolute point positions")
        #this can be optimized to half-matrix
        l = len(pointnames)
        mat = np.empty((l,l),np.float64)
        for i,pointi in enumerate(pointnames):
            for j,pointj in enumerate(pointnames):
                mat[i,j] = self.distance(pointi,pointj)
        return mat

    def neighbors(self, pointname):
        y,x = self[pointname]
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)] 
        return [self.pointnames[y, x] for y, x in neighbors if (0 <= x < self.nx) and (0 <= y < self.ny)]
                
        






