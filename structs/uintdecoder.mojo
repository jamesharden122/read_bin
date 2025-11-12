from collections import InlineArray
from .unsigned_int import Uint64TwosComp, Uint8TwosComp, Uint32TwosComp

struct Uint64_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 8], N]
    var n: Int
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 8], N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            print("file:",self.file_path)
            print("read file succesfully")
            buffer = file.read_bytes(-1)
            print(len(buffer))
            print("read bytes succesfully")
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer) // 8)
        for i in range(chunks):
            var base = i * 8
            tmp[i] = SIMD[DType.uint8, 8](
                Int(buffer[base + 0]), Int(buffer[base + 1]), Int(buffer[base + 2]), Int(buffer[base + 3]),
                Int(buffer[base + 4]), Int(buffer[base + 5]), Int(buffer[base + 6]), Int(buffer[base + 7])
            )
        for i in range(chunks, N):
            tmp[i] = SIMD[DType.uint8, 8](0,0,0,0,0,0,0,0)
        self.list_simd = tmp
        self.n = N

    fn bytes_to_uint(self) -> InlineArray[UInt64, N]:
        try:  
            var conv = Uint64TwosComp[N](self.list_simd)
            var out = conv.uint64_conversion()
            return out
        except:
            print("uint64 error")
            return InlineArray[UInt64, N](fill=0)


struct Uint32_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 4], N]
    var n: Int
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 4], N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            print("file:",self.file_path)
            print("read file succesfully")
            buffer = file.read_bytes(-1)
            print(len(buffer))
            print("read bytes succesfully")
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer) // 4)
        for i in range(chunks):
            var base = i * 4
            tmp[i] = SIMD[DType.uint8, 4](
                Int(buffer[base + 0]), Int(buffer[base + 1]), Int(buffer[base + 2]), Int(buffer[base + 3])
            )
        for i in range(chunks, N):
            tmp[i] = SIMD[DType.uint8, 4](0,0,0,0)
        self.list_simd = tmp
        self.n = N

    fn bytes_to_uint(self) -> InlineArray[UInt32, N]:
        try:
            var conv = Uint32TwosComp[N](self.list_simd)
            var out = conv.uint32_conversion()
            return out
        except:
            print("uint32 Error")
            return InlineArray[UInt32, N](fill=0)

struct Uint8_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 1], N]
    var n: Int
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 1], N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            print("file:",self.file_path)
            print("read file succesfully")
            buffer = file.read_bytes(-1)
            print(len(buffer))
            print("read bytes succesfully")
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer))
        for i in range(chunks):
            tmp[i] = SIMD[DType.uint8, 1](Int(buffer[i]))
        for i in range(chunks, N):
            tmp[i] = SIMD[DType.uint8, 1](0)
        self.list_simd = tmp
        self.n = N

    fn bytes_to_uint(self) -> InlineArray[UInt8, N]:
        try:
            var conv = Uint8TwosComp[N](self.list_simd)
            var out = conv.uint8_conversion()
            return out
        except:
            print("uint8 Error")
            return InlineArray[UInt8, N](fill=0)
