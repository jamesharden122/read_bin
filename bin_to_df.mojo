from .structs.floatdecoder import float64_column
from .structs.intdecoder import Int64_column, Int32_column
from .structs.uintdecoder import Uint64_column, Uint32_column, Uint8_column
from max.tensor import Tensor, TensorSpec, TensorShape
from max.graph.checkpoint import load, save, TensorDict
from utils.index import Index
from collections import Dict

struct DfReader:
    var bin_paths: Dict[String,List[String]] 
    var base_path: String

    fn __init__(out self, bin_paths: Dict[String,List[String]], base_path: String):
        self.bin_paths = bin_paths
        self.base_path = base_path
    
    def create(self) -> TensorDict:
        var tensors = TensorDict()
        for e in self.bin_paths.items():
            if (e[].key == "int64") & (len(e[].value) > 0):
                for nm in e[].value:                    
                    var temp = Int64_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor = temp.bytes_to_int()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())

            elif (e[].key == "int32") & (len(e[].value) > 0):
                for nm in e[].value:
                    var temp = Int32_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor = temp.bytes_to_int()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())

            elif (e[].key == "uint64") & (len(e[].value) > 0):
                for nm in e[].value:
                    var temp = Uint64_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor = temp.bytes_to_uint()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())
            elif (e[].key == "uint32") & (len(e[].value) > 0):
                for nm in e[].value:
                    var temp = Uint32_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor = temp.bytes_to_uint()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())
            elif (e[].key == "uint8") & (len(e[].value) > 0):
                for nm in e[].value:
                    var temp = Uint8_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor: Tensor[DType.uint8] = temp.bytes_to_uint()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())
            elif (e[].key == "f64") & (len(e[].value) > 0):
                for nm in e[].value:
                    var temp = float64_column(nm.__getitem__(),self.base_path.__add__(nm.__getitem__()))
                    var temp_tensor: Tensor[DType.float64] = temp.bytes_to_float()
                    print(temp_tensor)
                    tensors.set(nm.__getitem__(),temp_tensor)
                    print(nm.__getitem__())
        return(tensors)
        


