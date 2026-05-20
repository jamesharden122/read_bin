from .structs.floatdecoder import float64_column
from .structs.intdecoder import Int64_column, Int32_column
from .structs.uintdecoder import Uint64_column, Uint32_column, Uint8_column
from layout import Layout, LayoutTensor
from std.collections import Dict, InlineArray


struct DfReader[N: Int, L: Int]:
    var bin_paths: Dict[String, List[String]]
    var base_path: String
    var col_names: List[String]
    var data_storage: InlineArray[Float64, Self.N * Self.L]

    def __init__(out self, bin_paths: Dict[String, List[String]], base_path: String):
        self.bin_paths = bin_paths.copy()
        self.base_path = base_path
        self.col_names = List[String]()
        self.data_storage = InlineArray[Float64, Self.N * Self.L](fill=0.0)

    def create(mut self) -> LayoutTensor[DType.float64, Layout.row_major(Self.N, Self.L), MutAnyOrigin]:
        for idx in range(Self.N * Self.L):
            self.data_storage[idx] = 0.0
        var tensors = LayoutTensor[DType.float64, Layout.row_major(Self.N, Self.L)](self.data_storage)
        var i = 0
        for e in self.bin_paths.items():
            if (e.key == "int64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = Int64_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_int()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1

            elif (e.key == "int32") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = Int32_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_int()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1

            elif (e.key == "uint64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = Uint64_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1

            elif (e.key == "uint32") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = Uint32_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1

            elif (e.key == "uint8") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = Uint8_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_uint()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](Float64(vals[r])))
                    i += 1

            elif (e.key == "f64") and (len(e.value) > 0):
                for nm in e.value:
                    self.col_names.append(nm)
                    # print(self.base_path.__add__(nm))
                    var col = float64_column[Self.N](nm, self.base_path.__add__(nm))
                    var vals = col.bytes_to_float()
                    for r in range(Self.N):
                        tensors.aligned_store[1](r, i, SIMD[DType.float64, 1](vals[r]))
                    i += 1
        return tensors.as_any_origin()

    @staticmethod
    def preview(
        tensor: LayoutTensor[DType.float64, Layout.row_major(Self.N, Self.L), MutAnyOrigin],
        rows: Int,
        cols: Int,
    ):
        print("Tensor shape:", rows, "x", cols)
        var preview_rows = 15 if rows > 14 else rows
        for r in range(preview_rows):
            var line = ""
            for c in range(cols):
                var v = tensor.aligned_load[1](r, c)[0]
                line = line.__add__(String(v)).__add__(" ")
            print(line)

    # Static utility: pretty-print a small preview of the tensor
