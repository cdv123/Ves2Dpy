export MPI_HOME=/cosma/local/openmpi/gnu_11.1.0/4.1.1
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export PETSC_OPTIONS="-use_gpu_aware_mpi 0"
# mpirun -n 2 --mca pml ob1 --mca btl self,vader python distributed_driver_confined_petsc.py
