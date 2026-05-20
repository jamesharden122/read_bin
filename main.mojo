from std.collections import Dict
from read_bin.bin_to_df import DfReader


def main() raises:
    var base_path = "data/"
    var bin_paths = Dict[String, List[String]]()

    var uint64_cols = List[String]()
    uint64_cols.append("ts_recv.bin")
    bin_paths["uint64"] = uint64_cols^

    var uint32_cols = List[String]()
    uint32_cols.append("ts_in_delta.bin")
    uint32_cols.append("size.bin")
    bin_paths["uint32"] = uint32_cols^

    var int64_cols = List[String]()
    int64_cols.append("price.bin")
    bin_paths["int64"] = int64_cols^

    var uint8_cols = List[String]()
    uint8_cols.append("side.bin")
    bin_paths["uint8"] = uint8_cols^

    bin_paths["int32"] = List[String]()
    bin_paths["f64"] = List[String]()

    var df = DfReader[2354, 5](bin_paths, base_path)
    var tensor = df.create()

    print("Tensor shape:", 2354, "x", 5)
    for r in range(10):
        var line = ""
        for c in range(5):
            line = line.__add__(String(tensor.aligned_load[1](r, c)[0])).__add__(" ")
        print(line)
