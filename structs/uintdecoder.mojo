from std.collections import InlineArray, List
from .unsigned_int import Uint64TwosComp, Uint8TwosComp, Uint32TwosComp


struct Uint64_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 8], Self.N]
    var n: Int

    def __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 8], Self.N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            buffer = file.read_bytes(-1)
            print("file:", self.file_path, " ", len(buffer))
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer) // 8)
        if chunks > Self.N:
            chunks = Self.N
        for i in range(chunks):
            var base = i * 8
            tmp[i] = SIMD[DType.uint8, 8](
                UInt8(buffer[base + 0]),
                UInt8(buffer[base + 1]),
                UInt8(buffer[base + 2]),
                UInt8(buffer[base + 3]),
                UInt8(buffer[base + 4]),
                UInt8(buffer[base + 5]),
                UInt8(buffer[base + 6]),
                UInt8(buffer[base + 7]),
            )
        for i in range(chunks, Self.N):
            tmp[i] = SIMD[DType.uint8, 8](0, 0, 0, 0, 0, 0, 0, 0)
        self.list_simd = tmp
        self.n = Self.N

    def bytes_to_uint(self) -> InlineArray[UInt64, Self.N]:
        try:
            var conv = Uint64TwosComp[Self.N](self.list_simd)
            var out = conv.uint64_conversion()
            return out
        except:
            print("uint64 error")
            return InlineArray[UInt64, Self.N](fill=0)


struct Uint32_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 4], Self.N]
    var n: Int

    def __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 4], Self.N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            buffer = file.read_bytes(-1)
            print("file:", self.file_path, " ", len(buffer))
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer) // 4)
        if chunks > Self.N:
            chunks = Self.N
        for i in range(chunks):
            var base = i * 4
            tmp[i] = SIMD[DType.uint8, 4](UInt8(buffer[base + 0]), UInt8(buffer[base + 1]), UInt8(buffer[base + 2]), UInt8(buffer[base + 3]))
        for i in range(chunks, Self.N):
            tmp[i] = SIMD[DType.uint8, 4](0, 0, 0, 0)
        self.list_simd = tmp
        self.n = Self.N

    def bytes_to_uint(self) -> InlineArray[UInt32, Self.N]:
        try:
            var conv = Uint32TwosComp[Self.N](self.list_simd)
            var out = conv.uint32_conversion()
            return out
        except:
            print("uint32 Error")
            return InlineArray[UInt32, Self.N](fill=0)


struct Uint8_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 1], Self.N]
    var n: Int

    def __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 1], Self.N](uninitialized=True)
        var buffer: List[UInt8]
        try:
            var file = open(self.file_path, "r")
            buffer = file.read_bytes(-1)
            print("file:", self.file_path, " ", len(buffer))
            file.close()
        except:
            buffer = List[UInt8]()
        var chunks = Int(len(buffer))
        if chunks > Self.N:
            chunks = Self.N
        for i in range(chunks):
            tmp[i] = SIMD[DType.uint8, 1](UInt8(buffer[i]))
        for i in range(chunks, Self.N):
            tmp[i] = SIMD[DType.uint8, 1](0)
        self.list_simd = tmp
        self.n = Self.N

    def bytes_to_uint(self) -> InlineArray[UInt8, Self.N]:
        try:
            var conv = Uint8TwosComp[Self.N](self.list_simd)
            var out = conv.uint8_conversion()
            return out
        except:
            print("uint8 Error")
            return InlineArray[UInt8, Self.N](fill=0)
