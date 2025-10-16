from max.tensor import Tensor, TensorSpec, TensorShape
from .twos_comp import Int32TwosComp, Int64TwosComp

struct Int64_column:
    var name: String
    var file_path: String
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path

    fn bytes_to_int(self) -> Tensor[DType.int64]:
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
            var int_struct = Int64TwosComp(list_simd[1:])
            var data = int_struct.int64_conversion()
            return(data)
        except:
            print("Int64 Error")
            return(Tensor[DType.int64]())


struct Int32_column:
    var name: String
    var file_path: String
    fn __init__(out self, name: String, file_path: String):
        self.name = name
        self.file_path = file_path

    fn bytes_to_int(self) -> Tensor[DType.int32]:
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
            var int_struct = Int32TwosComp(list_simd[2:])
            var data = int_struct.int32_conversion()
            return(data)
        except:
            print("Int32 Error")
            return(Tensor[DType.int32]())