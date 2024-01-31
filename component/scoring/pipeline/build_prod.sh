kernel_name="rapids"
echo "--- Install jupyter notebook kernel: ${kernel_name} ---"
conda create -n ${kernel_name} python=3.8.15 -y
source activate ${kernel_name}
conda install -y ipykernel
ipython kernel install --user --name=${kernel_name}

echo "Running: scoring_prod.py $@"
python scoring_prod.py $@