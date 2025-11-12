from collections import Dict
from read_bin.bin_to_df import DfReader
from utils.index import Index




def main():
    base_path = "./data/"
    var bin_paths = Dict[String,List[String]]()
    bin_paths["uint64"] = List[String]("ts_recv.bin")
    bin_paths["uint32"] = List[String]("ts_in_delta.bin", "size.bin")#"instrument_id.bin"
    bin_paths["uint8"] = List[String]()#"flags.bin","depth.bin"
    bin_paths["int64"] = List[String]("price.bin")
    bin_paths["int32"] = List[String]("ts_in_delta.bin")
    bin_paths["f64"] = List[String]()
    var df = DfReader[2355,5](bin_paths, base_path)
    var tensor = df.create()
    var rows = 2355
    var cols = 5
    print("Tensor shape:", rows, "x", cols)
    var preview_rows = 5 if rows > 5 else rows
    for r in range(preview_rows):
        var line = ""
        for c in range(cols):
            var v = tensor.aligned_load[1](r, c)[0]
            line = line.__add__(v.__str__()).__add__(" ")
        print(line)

    # Extract tensors from TensorDict and stack them
    #var input_keys = List[String]("ts_in_delta", "size", "price")
    #var tensors = [tens_dict[k] for k in input_keys]

    #var num_rows = tensors[0].shape[0]
    #var num_features = tensors.size
    #var data = InlineArray[Float32]

    #for i in range(num_features):
    #    var col_tensor = tensors[i]
    #    for j in range(num_rows):
    #        input_tensor[Index(j, i)] = col_tensor[Index(j, 0)]
