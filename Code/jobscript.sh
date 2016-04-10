
module load mkl/intel/psxe2015/mklvars
module load apps/caffe

~/anaconda/bin/python ~/Code/model.py -epochs 1 > ~/Code/output.txt 2> ~/Code/errorout.txt