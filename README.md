# Stokes Periodization

This repository contains code for 1D periodization of the Stokes problem on the unit box \([0,1]^3\) by adding images of the domain.

## Requirements

- C++17+ compiler with OpenMP
- `mpirun` (e.g., OpenMPI or MPICH)
- Autotools (for building PVFMM)

## Installation & Usage

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/dmalhotra/stokes-periodize.git
cd stokes-periodize/extern/pvfmm

# Build PVFMM
./autogen.sh
./configure CXXFLAGS="-march=native -O3"
make -j

# Build the main executable
cd ../..
make bin/test1

# Run the test (set threads as needed)
export OMP_NUM_THREADS=16
time mpirun -n 1 --map-by slot:pe=$OMP_NUM_THREADS ./bin/test1
```
