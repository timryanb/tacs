import os
import tacs
import mpi4py

tacs_dir = os.path.dirname(tacs.__file__)
mpi4py_dir = os.path.dirname(mpi4py.__file__)

tacs_files = os.listdir(tacs_dir)
mpi4py_files = os.listdir(mpi4py_dir)

print("TACS LIBRARIES...............................")
for file in tacs_files:
    if os.path.splitext(file)[-1] == '.so':
        os.system(f"otool -L {tacs_dir}/{file}")

print("MPI4PY LIBRARIES...............................")
for file in mpi4py_files:
    if os.path.splitext(file)[-1] == '.so':
        os.system(f"otool -L {mpi4py_dir}/{file}")