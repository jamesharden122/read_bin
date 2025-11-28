from collections import Dict
from read_bin.bin_to_df import DfReader
from utils.index import Index




def main():
    # Update the data folder to the quant_models_expl tmp_data directory
    # Note: this path is relative to this module (mojo_modules/read_bin)
    #       adjust if your layout changes, but keep trailing slash
    base_path = "../../quant_models_expl/tmp_data/"

    var bin_paths = Dict[String, List[String]]()
    # Only two dtypes are used here per request:
    # - int32: date.bin
    # - f64  : all other .bin files in the listing
    bin_paths["int32"] = List[String](
        "date.bin"
    )

    bin_paths["f64"] = List[String](
        "aus_portret.bin",   "aus_portretx.bin",
        "aut_portret.bin",   "aut_portretx.bin",
        "bel_portret.bin",   "bel_portretx.bin",
        "bra_portret.bin",   "bra_portretx.bin",
        "che_portret.bin",   "che_portretx.bin",
        "chl_portret.bin",   "chl_portretx.bin",
        "chn_portret.bin",   "chn_portretx.bin",
        "col_portret.bin",   "col_portretx.bin",
        "deu_portret.bin",   "deu_portretx.bin",
        "dnk_portret.bin",   "dnk_portretx.bin",
        "egy_portret.bin",   "egy_portretx.bin",
        "esp_portret.bin",   "esp_portretx.bin",
        "fin_portret.bin",   "fin_portretx.bin",
        "fra_portret.bin",   "fra_portretx.bin",
        "gbr_portret.bin",   "gbr_portretx.bin",
        "grc_portret.bin",   "grc_portretx.bin",
        "hkg_portret.bin",   "hkg_portretx.bin",
        "hun_portret.bin",   "hun_portretx.bin",
        "idn_portret.bin",   "idn_portretx.bin",
        "ind_portret.bin",   "ind_portretx.bin",
        "irl_portret.bin",   "irl_portretx.bin",
        "ita_portret.bin",   "ita_portretx.bin",
        "jpn_portret.bin",   "jpn_portretx.bin",
        "kor_portret.bin",   "kor_portretx.bin",
        "mex_portret.bin",   "mex_portretx.bin",
        "mys_portret.bin",   "mys_portretx.bin",
        "nld_portret.bin",   "nld_portretx.bin",
        "nor_portret.bin",   "nor_portretx.bin",
        "nzl_portret.bin",   "nzl_portretx.bin",
        "phl_portret.bin",   "phl_portretx.bin",
        "pol_portret.bin",   "pol_portretx.bin",
        "prt_portret.bin",   "prt_portretx.bin",
        "sgp_portret.bin",   "sgp_portretx.bin",
        "swe_portret.bin",   "swe_portretx.bin",
        "tha_portret.bin",   "tha_portretx.bin",
        "tur_portret.bin",   "tur_portretx.bin",
        "twn_portret.bin",   "twn_portretx.bin",
        "zaf_portret.bin",   "zaf_portretx.bin",
        "ewretd.bin",        "ewretx.bin",
        "vwretd.bin",        "vwretx.bin",
        "sprtrn.bin"
    )

    # Keep other dtypes defined but empty for clarity
    bin_paths["uint64"] = List[String]()
    bin_paths["uint32"] = List[String]()
    bin_paths["uint8"]  = List[String]()
    bin_paths["int64"]  = List[String]()

    # Compile-time sizes: choose N large enough for your files, and L matching total columns
    # Here: L = 1 (date) + 81 float columns = 82
    var df = DfReader[1437, 82](bin_paths, base_path)
    var tensor = df.create()

    var rows = 4096
    var cols = 5
    print("Tensor shape:", rows, "x", cols)
    var preview_rows = 15 if rows > 14 else rows
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
