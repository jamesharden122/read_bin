from std.collections import InlineArray, List
from .twos_comp import Int32TwosComp, Int64TwosComp
from std.memory import bitcast


struct Int64_column[N: Int]:
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

    def bytes_to_int(self) -> InlineArray[Int64, Self.N]:
        try:
            var conv = Int64TwosComp[Self.N](self.list_simd)
            return conv.int64_conversion()
        except:
            print("Int64 Error")
            return InlineArray[Int64, Self.N](fill=0)


struct Int32_column[N: Int]:
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

    def bytes_to_int(self) -> InlineArray[Int32, Self.N]:
        var out = InlineArray[Int32, Self.N](uninitialized=True)
        for i in range(Self.N):
            var bytes = self.list_simd[i]
            var little_endian_bytes = SIMD[DType.uint8, 4](bytes[3], bytes[2], bytes[1], bytes[0])
            out[i] = bitcast[DType.int32, 1](little_endian_bytes)[0]
        return out
