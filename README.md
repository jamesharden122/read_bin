# read_bin — Binary-to-Tensor Utilities (Mojo + MAX)

`read_bin` loads fixed‑width binary columns (int32/64, uint8/32/64, float64) and produces an `N × L` dense tensor for downstream processing. It uses compile‑time generics, SIMD bit manipulation, and `LayoutTensor` streaming stores for speed and predictability.

## Features
- Per‑dtype decoders under `structs/` (two’s‑complement and IEEE‑754).
- Compile‑time generics `[N: Int]` for row count (and `[L: Int]` for column count in `DfReader`).
- Zero‑fill when binaries are shorter than `N`; truncation when longer.
- Column‑wise population of an `N × L` `LayoutTensor[DType.float64]` via `aligned_store`.

## Layout
- `main.mojo`: Example runner configuring files and invoking `DfReader[N, L]`.
- `bin_to_df.mojo`: Orchestrates decoding and builds the `N × L` tensor via aligned stores.
- `structs/`
  - `intdecoder.mojo`, `uintdecoder.mojo`, `floatdecoder.mojo`: File readers returning `InlineArray` values.
  - `twos_comp.mojo`, `unsigned_int.mojo`: SIMD bit‑to‑integer conversion.
  - `ieee754.mojo`: SIMD bit‑to‑float64 conversion.

## Prerequisites
- Mojo + MAX stdlib available via Pixi.
- Recommended Pixi config in `read_bin/pixi.toml` (you may need Modular’s MAX channel/auth):
  - Channels: `https://repo.modular.com/max`, `conda-forge`
  - Authenticate if required: `pixi auth add modular https://repo.modular.com --username <email> --password <token>`

## Quick Start
1) Enter the environment and run the example

```
cd read_bin
pixi install
pixi run mojo main.mojo
```

2) Configure inputs in `main.mojo`

- `base_path`: directory containing your binaries (must end with `/`).
- `bin_paths`: map dtype → list of filenames, e.g.

```
bin_paths["uint64"] = List[String]("ts_recv.bin")
bin_paths["uint32"] = List[String]("ts_in_delta.bin", "size.bin")
bin_paths["int64"]  = List[String]("price.bin")
```

3) Choose compile‑time sizes

```
var df = DfReader[ROWS, COLS](bin_paths, base_path)
var tensor = df.create()  // LayoutTensor[DType.float64, Layout.row_major(ROWS, COLS)]
```

- `ROWS` (N): rows decoded per column. If a file encodes fewer than `ROWS`, remaining entries are zero‑filled; if more, excess is ignored.
- `COLS` (L): number of columns you plan to populate (sum of all lists in `bin_paths`).

## How It Works
- Readers (`*decoder.mojo`) open files in binary mode, slice into fixed‑width chunks, and build `InlineArray[SIMD[uint8, W], N]`.
- Converters (`twos_comp`/`unsigned_int`/`ieee754`) map bits to numbers using SIMD and power‑of‑two masks.
- `DfReader.create()` converts each column to `Float64` and stores it with:

```
tensors.aligned_store[1](row, col, SIMD[DType.float64, 1](value))
```

This streams values into the `N × L` `LayoutTensor` with aligned writes.

## Minimal API Examples
- Int64 column
```
from read_bin.structs.intdecoder import Int64_column
var col = Int64_column[1024]("price.bin", "/path/")
var vals = col.bytes_to_int()           # InlineArray[Int64, 1024]
```

- Float64 column
```
from read_bin.structs.floatdecoder import float64_column
var col = float64_column[1024]("price.bin", "/path/")
var vals = col.bytes_to_float()         # InlineArray[Float64, 1024]
```

- Build N×L tensor and read values
```
from read_bin.bin_to_df import DfReader
var df = DfReader[1024, 3](bin_paths, base_path)
var tensor = df.create()                # LayoutTensor[float64, row_major(1024, 3)]
# Read first row values across 3 columns
var r = 0
for c in range(3):
    print(tensor.aligned_load[1](r, c)[0])
```

## Tips & Gotchas
- Compile‑time generics: `N` and `L` must be constants at compile time.
- File lengths: choose `N` so `N * element_width` matches your files (or accept zero‑fill/truncation). Widths: uint8=1, int32/uint32=4, int64/uint64/float64=8 bytes.
- Endianness: current decoders interpret bytes as provided (no byte‑order swap). Ensure your binaries’ order matches expectations.
- Paths: `base_path + filename` is used; ensure `base_path` ends with a `/`.
- Performance: `aligned_store` uses width 1 by default. You can batch rows (width 2/4/8) if your data and alignment allow.

## Extending
- Add new dtypes by mirroring the decoder + converter pattern under `structs/`.
- For vectorized stores, group `width` consecutive rows into `SIMD[DType.float64, width]` and call `aligned_store[width](row, col, simd_vals)`.

## Troubleshooting
- “unable to locate module …”: verify Pixi channels and authentication for MAX.
- “failed to infer parameter 'mut'”: avoid typing `LayoutTensor[...]` as a struct field; initialize tensors inside functions and return them.
- String slice issues in bit parsing: pass single‑char strings to helpers (done via `.__getitem__(0)`).

## License
Add the appropriate license for your repository.
