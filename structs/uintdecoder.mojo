from .unsigned_int import Int64TwosComp, Uint64TwosComp, Uint8TwosComp

struct Uint64_column:
    var name: String
    var file_path: String
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path

    fn bytes_to_uint(self) -> Tensor[DType.uint64]:
        try:
            print(self.file_path)
            var file =open(self.file_path,"r")
            var buffer = file.read_bytes(-1)
            file.close()
            var list_simd = List[SIMD[DType.uint8,8]]()
            for i in range(0,len(buffer),8):
                var temp_simd = SIMD[DType.uint8,8](
                    Int(buffer[i]),
                    Int(buffer[i+1]),
                    Int(buffer[i+2]),
                    Int(buffer[i+3]),
                    Int(buffer[i+4]),
                    Int(buffer[i+5]),
                    Int(buffer[i+6]),
                    Int(buffer[i+7])
                )
                list_simd.append(temp_simd)
            print(len(list_simd))
            var float_struct = Uint64TwosComp(list_simd[1:])
            var data = float_struct.uint64_conversion()
            return(data)
        except:
            print("uint64 error")
            return(Tensor[DType.uint64]())


struct Uint32_column:
    var name: String
    var file_path: String
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path

    fn bytes_to_uint(self) -> Tensor[DType.uint32]:
        try:
            print(self.file_path)
            var file =open(self.file_path,"r")
            var buffer = file.read_bytes(-1)
            file.close()
            var list_simd = List[SIMD[DType.uint8,4]]()
            for i in range(0,len(buffer),4):
                var temp_simd = SIMD[DType.uint8,4](
                    Int(buffer[i]),
                    Int(buffer[i+1]),
                    Int(buffer[i+2]),
                    Int(buffer[i+3])
                )
                list_simd.append(temp_simd)
            print(len(list_simd))
            var float_struct = Uint32TwosComp(list_simd[2:])
            var data = float_struct.uint32_conversion()
            return(data)
        except:
            print("uint32 Error")
            return(Tensor[DType.uint32]())

struct Uint8_column:
    var name: String
    var file_path: String
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path

    fn bytes_to_uint(self) -> Tensor[DType.uint8]:
        try:
            print(self.file_path)
            var file =open(self.file_path,"r")
            var buffer = file.read_bytes(-1)
            file.close()
            var list_simd = List[SIMD[DType.uint8,1]]()
            for i in range(0,len(buffer)):
                var temp_simd = SIMD[DType.uint8,1](
                    Int(buffer[i])
                )
                list_simd.append(temp_simd)
            print(len(list_simd))
            var float_struct = Uint8TwosComp(list_simd)
            var data = float_struct.uint8_conversion()
            return(data)
        except:
            print("uint8 Error")
            return(Tensor[DType.uint8]())