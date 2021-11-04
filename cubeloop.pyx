import cython

def cubeloop(int[:,:] coords, int nz,double[:,:,:] cube,double[:,:,:] vcube,double[:] spec,double[:] espec):

    cdef int cnum=coords.shape[0]
    cdef int dnum=cube.shape[0]
    cdef int d,q

    # coords is a 2 column variable row array

    for d in range(dnum):
        for q in range(cnum):
            spec[d]+=cube[d,coords[q][0],coords[q][1]]
            espec[d]+=vcube[d,coords[q][0],coords[q][1]]
    return spec, espec