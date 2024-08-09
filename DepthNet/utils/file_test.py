datafile = "../datasets/eigen_train_files_with_gt_dense.txt"

with open(datafile, 'r') as f:
    fileset = f.readlines()
fileset = sorted(fileset)

print(fileset[0].split())