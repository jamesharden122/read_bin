from .structs.floatdecoder import float64_column
from .structs.intdecoder import Int64_column, Int32_column
from .structs.uintdecoder import Uint64_column, Uint32_column, Uint8_column
from layout import Layout, LayoutTensor
from collections import Dict, InlineArray

struct DfReader[N: Int,L: Int]:
    var bin_paths: Dict[String, List[String]]
    var base_path: String
    var col_names: List[String]

    fn __init__(out self, bin_paths: Dict[String, List[String]], base_path: String):
        self.bin_paths = bin_paths.copy()
        self.base_path = base_path
        self.col_names: List[String] = []


    fn create(mut self) -> LayoutTensor[mut=True,DType.float64, Layout.row_major(N, L), MutableAnyOrigin]:
        var uninit_data = InlineArray[Float64,N*L](fill=0.0)
        var tensors = LayoutTensor[DType.float64, Layout.row_major(N, L), MutableAnyOrigin](uninit_data)
        var i = 0
        for e in self.bin_paths.items():
            if (e.key == "int64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = Int64_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_int()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1
                    
            elif (e.key == "int32") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = Int32_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_int()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1
                                        
            elif (e.key == "uint64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = Uint64_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1
                
                    
            elif (e.key == "uint32") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = Uint32_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1
                   
            elif (e.key == "uint8") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = Uint8_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1
                    
            elif (e.key == "f64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    print(self.base_path.__add__(nm))
                    var col = float64_column[N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_float()
                    for r in range(N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](vals[r]))
        return tensors
                    
