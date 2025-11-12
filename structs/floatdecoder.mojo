from .ieee754 import IEEE754Processor
from collections import InlineArray


# Struct that reads bytes from a file and converts to InlineArray[Float64, N]
struct float64_column[N: Int]:
    var name: String
    var file_path: String
    var list_simd: InlineArray[SIMD[DType.uint8, 8], N]
    var n: Int


    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path
        var tmp = InlineArray[SIMD[DType.uint8, 8], N](uninitialized=True)
        try:
            var file = open(self.file_path, "r")
            print("file:",self.file_path)
            print("read file succesfully")
            buffer = file.read_bytes(-1)
            print(len(buffer))
            print("read bytes succesfully")
            file.close()
            var chunks = Int(len(buffer) // 8)
            for i in range(chunks):
                var base = i * 8
                tmp[i] = SIMD[DType.uint8, 8](
                    Int(buffer[base + 0]), Int(buffer[base + 1]), Int(buffer[base + 2]), Int(buffer[base + 3]),
                    Int(buffer[base + 4]), Int(buffer[base + 5]), Int(buffer[base + 6]), Int(buffer[base + 7])
                )
            # Zero-initialize any remaining lanes if file shorter than N*8 bytes
            for i in range(chunks, N):
                tmp[i] = SIMD[DType.uint8, 8](0,0,0,0,0,0,0,0)
        except:
            for i in range(N):
                tmp[i] = SIMD[DType.uint8, 8](0,0,0,0,0,0,0,0)
        self.list_simd = tmp
        self.n = N


    fn bytes_to_float(self) -> InlineArray[Float64, N]:
        var float_struct = IEEE754Processor[N](self.list_simd)
        return float_struct.float_64_conversion()
