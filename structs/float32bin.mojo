from std.collections import List
from std.memory import bitcast


struct Float32BinaryWriter(Movable):
    var bytes: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()

    def append_u8(mut self, value: UInt8):
        self.bytes.append(value)

    def append_i32(mut self, value: Int):
        var packed = bitcast[DType.uint8, 4](SIMD[DType.int32, 1](Int32(value)))
        for idx in range(4):
            self.bytes.append(UInt8(packed[idx]))

    def append_f32(mut self, value: Float32):
        var packed = bitcast[DType.uint8, 4](SIMD[DType.float32, 1](value))
        for idx in range(4):
            self.bytes.append(UInt8(packed[idx]))

    def write(mut self, filepath: String) raises:
        var file = open(filepath, "w")
        file.write_bytes(self.bytes)
        file.close()


struct Float32BinaryReader(Movable):
    var bytes: List[UInt8]

    def __init__(out self, filepath: String) raises:
        self.bytes = List[UInt8]()
        var file = open(filepath, "r")
        var data = file.read_bytes(-1)
        file.close()
        for idx in range(len(data)):
            self.bytes.append(UInt8(data[idx]))

    def byte_length(self) -> Int:
        return len(self.bytes)

    def read_u8(self, offset: Int) raises -> UInt8:
        if offset < 0 or offset >= len(self.bytes):
            raise Error("invalid binary file: truncated UInt8")
        return self.bytes[offset]

    def read_i32(self, offset: Int) raises -> Int:
        if offset < 0 or offset + 4 > len(self.bytes):
            raise Error("invalid binary file: truncated Int32")
        var packed = SIMD[DType.uint8, 4](
            self.bytes[offset + 0],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        )
        var unpacked = bitcast[DType.int32, 1](packed)
        return Int(unpacked[0])

    def read_f32(self, offset: Int) raises -> Float32:
        if offset < 0 or offset + 4 > len(self.bytes):
            raise Error("invalid binary file: truncated Float32")
        var packed = SIMD[DType.uint8, 4](
            self.bytes[offset + 0],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        )
        var unpacked = bitcast[DType.float32, 1](packed)
        return unpacked[0]
