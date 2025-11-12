from collections import InlineArray
from .twos_comp import Int32TwosComp, Int64TwosComp

struct Int64_column[N: Int]:
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

    fn bytes_to_int(self) -> InlineArray[Int64, N]:
        try:
            var out = InlineArray[Int64, N](uninitialized=True)
            var conv = Int64TwosComp[N](self.list_simd)
            var bit_lists = conv.process_simd_list(self.list_simd)
            for i in range(N):
                out[i] = conv.binary_to_int(bit_lists[i])[0]
            return out
        except:
            print("Int64 Error")
            return InlineArray[Int64, N](fill=0)


struct Int32_column[N: Int]:
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

    fn bytes_to_int(self) -> InlineArray[Int32, N]:
        try:
            var out = InlineArray[Int32, N](uninitialized=True)
            var conv = Int32TwosComp[N](self.list_simd)
            var bit_lists = conv.process_simd_list(self.list_simd)
            for i in range(N):
                out[i] = conv.binary_to_int(bit_lists[i])[0]
            return out
        except:
            print("Int32 Error")
            return InlineArray[Int32, N](fill=0)
