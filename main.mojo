from max.tensor import Tensor, TensorSpec, TensorShape
from max.graph.checkpoint import load, save, TensorDict
from collections import Dict
from bin_to_df import DfReader
from max import nn
from utils.index import Index

def dense_layer(input: Tensor[DType.float32],
                 weights: Tensor[DType.float32],
                 bias: Tensor[DType.float32]) -> Tensor[DType.float32]:
    return relu(matmul(input, weights) + bias)



def main():
    base_path = "/home/yakaman/Dropbox/Desktop/trading/bento_queries/"
    var bin_paths = Dict[String,List[String]]()
    bin_paths["uint64"] = List[String]("ts_recv.bin")
    bin_paths["uint32"] = List[String]("ts_in_delta.bin", "size.bin")#"instrument_id.bin"
    bin_paths["uint8"] = List[String]()#"flags.bin","depth.bin"
    bin_paths["int64"] = List[String]("price.bin")
    bin_paths["int32"] = List[String]("ts_in_delta.bin")
    bin_paths["f64"] = List[String]()
    var df = DfReader(bin_paths,base_path)
    var tens_dict = df.create()

    # Extract tensors from TensorDict and stack them
    var input_keys = List[String]("ts_in_delta", "size", "price")
    var tensors = [tens_dict[k] for k in input_keys]

    var num_rows = tensors[0].shape[0]
    var num_features = tensors.size
    var input_tensor = Tensor[DType.float32](TensorShape(num_rows, num_features))

    for i in range(num_features):
        var col_tensor = tensors[i]
        for j in range(num_rows):
            input_tensor[Index(j, i)] = col_tensor[Index(j, 0)]

    # Define weights for two dense layers manually
    var w1 = Tensor[DType.float32](TensorShape(num_features, 8))
    var b1 = Tensor[DType.float32](TensorShape(1, 8))
    var w2 = Tensor[DType.float32](TensorShape(8, 1))
    var b2 = Tensor[DType.float32](TensorShape(1, 1))

    w1.fill(0.01)
    b1.fill(0.0)
    w2.fill(0.01)
    b2.fill(0.0)

    var hidden = dense_layer(input_tensor, w1, b1)
    var output = matmul(hidden, w2) + b2

    print("First 5 outputs:")
    for i in range(min(5, output.shape[0])):
        print(output[Index(i, 0)])


