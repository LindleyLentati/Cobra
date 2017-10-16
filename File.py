import numpy as np
import ctypes as C

# convert between numpy dtypes and ctypes types.
nptypes_to_ctypes = {"|b1":C.c_bool,
                     "|S1":C.c_char,
                     "<i1":C.c_byte,
                     "<u1":C.c_ubyte,
                     "<i2":C.c_short,
                     "<u2":C.c_ushort,
                     "<i4":C.c_int,
                     "<u4":C.c_uint,
                     "<i8":C.c_long,
                     "<u8":C.c_ulong,
                     "<f4":C.c_float,
                     "<f8":C.c_double}

ctypes_to_nptypes = dict(zip(nptypes_to_ctypes.values(), nptypes_to_ctypes.keys()))

nbits_to_ctypes = {1:C.c_ubyte,
                   2:C.c_ubyte,
                   4:C.c_ubyte,
                   8:C.c_ubyte,
                   16:C.c_short,
                   32:C.c_float}

ctypes_to_nbits = dict(zip(nbits_to_ctypes.values(), nbits_to_ctypes.keys()))

nbits_to_dtype = {1:"<u1",
                  2:"<u1",
                  4:"<u1",
                  8:"<u1",
                  16:"<u2",
                  32:"<f4"}


class File(file):

    def __init__(self,filename,mode,nbits=8):
        file.__init__(self,filename,mode)
        self.nbits = nbits
        self.dtype = nbits_to_dtype[self.nbits]
        if nbits in [1,2,4]:
            self.bitfact = nbits/8.
            self.unpack = True
        else:
            self.bitfact = 1
            self.unpack = False


    def __del__(self):
        self.close()




def readDat(filename):
    f = File(filename,"r",nbits=32)
    data = np.fromfile(f,dtype="float32")
    return data
