from .ieee754 import IEEE754Processor
from std.collections import InlineArray
from std.memory import bitcast


# Struct that reads bytes from a file and converts to InlineArray[Float64, N]
struct float64_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 8], Self.N]
    var n: Int

    def __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 8], Self.N](uninitialized=True)
        try:
            var file = open(self.file_path, "r")
            buffer = file.read_bytes(-1)
            print("file:", self.file_path, " ", len(buffer))
            file.close()
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
            # Zero-initialize any remaining lanes if file shorter than N*8 bytes
            for i in range(chunks, Self.N):
                tmp[i] = SIMD[DType.uint8, 8](0, 0, 0, 0, 0, 0, 0, 0)
        except:
            for i in range(Self.N):
                tmp[i] = SIMD[DType.uint8, 8](0, 0, 0, 0, 0, 0, 0, 0)
        self.list_simd = tmp
        self.n = Self.N

    def bytes_to_float(self) -> InlineArray[Float64, Self.N]:
        var out = InlineArray[Float64, Self.N](uninitialized=True)
        for i in range(Self.N):
            var bytes = self.list_simd[i]
            var little_endian_bytes = SIMD[DType.uint8, 8](bytes[7], bytes[6], bytes[5], bytes[4], bytes[3], bytes[2], bytes[1], bytes[0])
            out[i] = bitcast[DType.float64, 1](little_endian_bytes)[0]
        return out
