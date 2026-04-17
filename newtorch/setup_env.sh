g++ -print-file-name=libstdc++.so.6
strings $(g++ -print-file-name=libstdc++.so.6) | grep GLIBCXX_3.4.31
export GCC_LIBDIR=$(dirname $(g++ -print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH=$GCC_LIBDIR:$LD_LIBRARY_PATH
torchrun --standalone --nnodes=1 --nproc-per-node=2 distributed_driver_confined_petsc.py
module purge
module load gnu_cmo
module load gnu_comp
module openmpi
module load openmpi
export GCC_LIBDIR=$(dirname $(g++ -print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH=$GCC_LIBDIR:$LD_LIBRARY_PATH
torchrun --standalone --nnodes=1 --nproc-per-node=2 distributed_driver_confined_petsc.py
cd vesicle-fork/Ves2Dpy/newtorch/
source /cosma/apps/do022/dc-dubo2/cuda-env/bin/activate
module load gnu_comp  openmpi
torchrun --standalone --nnodes=1 --nproc-per-node=2 distributed_driver_confined_petsc.py
MPI_HOME=$(dirname $(dirname $(which mpicc)))
export LD_LIB
