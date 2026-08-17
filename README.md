# FPGA-Implementation-of-a-Real-time-Multi-channel-Neural-Spike-Signal-Compression-System
Hardware (Verilog) &amp; Software (Python) Code 
# Real-Time Multi-Channel Neural Spike Signal Compression System

FPGA implementation and software validation code for a real-time, multi-channel
neural spike signal compression system integrating **spike detection (SD)** with
**compressed sensing (CS)**.

This repository accompanies the manuscript

> R. Zheng et al., "FPGA Implementation of a Real-time Multi-channel Neural Spike
> Signal Compression System," *IEEE Transactions on Biomedical Circuits and
> Systems* (under review, manuscript ID TBioCAS-2025-Dec-0436-Reg).

## Key Features

- **Single-threshold spike detection (STD)** with an intercept spike segment (ISS)
  technique that captures complete 100-sample spike windows.
- **Minimum Manhattan Distance Cluster-based (MDC) sensing matrix** with a
  **closed-form** derivation of the clustering parameter σ, enabling single-step
  matrix generation without iterative search.
- Hardware-friendly co-optimization: compressed dimension `M = 16 = 2^4` (division
  by shift) and `p(M−1) ∈ ℤ` (no multipliers).
- **384-channel** real-time compression on a Xilinx VCU129 FPGA at 50 MHz, with an
  average compression time of 8.12 μs per spike segment.
- Full software validation pipeline: IRLS reconstruction, PCA + K-means spike
  sorting, Kilosort4, and SNDR / CR / F1 / ARI evaluation.

## Repository Structure

```
.
├── hardware/                  # Verilog RTL (Xilinx Vivado)
│   ├── rtl/                   # Synthesizable design modules
│   │   ├── TOP_SD_CS.v        # Top-level integration
│   │   ├── TOP_module.v       # Module-level top
│   │   ├── SD_STD.v           # STD spike detection
│   │   ├── CS_MDC.v           # MDC-based CS compression
│   │   ├── Accumulator_100.v  # 100-sample accumulator
│   │   ├── BRAM.v             # Block RAM
│   │   ├── MEM_3data.v        # Data buffer
│   │   └── spi_module.v       # SPI interface
│   ├── testbench/
│   │   └── testbench.v        # Simulation testbench
│   └── constraints/
│       └── Master.xdc         # XDC constraints (VCU129)
├── software/                  # Python validation & analysis scripts
│   ├── SD_CS_SingleChannel_Sim.py      # Single-channel (Quiroga) reconstruction + spike sorting
│   ├── SD_CS_MultiChannel_Sim.py       # Multi-channel (Cortex Lab) reconstruction + spike sorting
│   ├── SD_CS_MultiChannel_CR60_Sim.py  # Multi-channel CR-sweep variant
│   ├── SD_CS_SpikeDetection.py         # STD detection-accuracy verification
│   ├── SD_vs_NEO_Comparison.py         # STD vs NEO detection comparison
│   ├── SensingMatrix_Comparison.py     # Sensing-matrix SNDR comparison (Table I)
│   ├── SD_CS_MDC_Comparison.py         # Iterative vs closed-form σ comparison (Table II)
│   ├── SD_CS_Parameter_Analysis.py     # Scaling-parameter p analysis
│   ├── SD_CS_Parameter_Analysis_Plot.py
│   ├── run_kilosort4_low_thresh.py     # Kilosort4 ground-truth spike sorting
│   ├── analyze_channels.py             # Channel analysis helper
│   └── build_recon_bin.py              # Reconstruction binary builder
└── data/                      # Dataset acquisition instructions (see data/README.md)
```

## Hardware (Vivado)

The RTL in `hardware/rtl/` is written in Verilog and targets a Xilinx VCU129
platform. To rebuild the project:

1. Open Vivado and create a new project targeting the Xilinx VCU129 (or compatible
   Ultrascale+ device).
2. Add the modules in `hardware/rtl/` as design sources and
   `hardware/testbench/testbench.v` as a simulation source.
3. Add `hardware/constraints/Master.xdc` as a constraints file.
4. Run behavioral simulation or synthesis/implementation as needed.

## Software (Python)

### Dependencies

```bash
pip install -r requirements.txt
```

For Kilosort4-based spike sorting (`run_kilosort4_low_thresh.py`), additionally
install [Kilosort4](https://github.com/MouseLand/Kilosort) and a compatible
PyTorch/CUDA environment (CPU-only execution is possible but slow).

### Datasets

The scripts read raw datasets from a data directory, which can be specified either
by placing files under `data/` in this repository or by setting the environment
variable `SPIKE_DATA_DIR`:

```bash
# Linux / macOS
export SPIKE_DATA_DIR=/path/to/datasets
# Windows PowerShell
$env:SPIKE_DATA_DIR = "D:\path\to\datasets"
```

See [`data/README.md`](data/README.md) for the required files and download links
(Quiroga simulated dataset and Cortex Lab Neuropixels 1.0 dataset).

### Typical workflow

1. `SD_CS_SingleChannel_Sim.py` — single-channel reconstruction, F1/ARI, figures
   (Quiroga, four difficulty levels).
2. `SD_CS_MultiChannel_Sim.py` — multi-channel reconstruction and spike sorting
   (Cortex Lab, 384 channels, 212 ground-truth clusters).
3. `SD_vs_NEO_Comparison.py` and `SensingMatrix_Comparison.py` — detector and
   sensing-matrix comparisons used in Tables I and the detection analysis.
4. `SD_CS_MDC_Comparison.py` — compares the proposed closed-form σ generation with
   iterative search (Table II).
5. `run_kilosort4_low_thresh.py` — Kilosort4 ground-truth labels for the
   multi-channel F1 evaluation.

## Citation

If you use this code in your research, please cite the manuscript above (DOI to be
added upon publication).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## Contact

For questions about the code, please open an issue in this repository or contact
the corresponding author.
