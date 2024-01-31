build_json="build.json"

package_versions=$(jq -r '.package_versions | to_entries[] | "\(.key)==\(.value)"' "$build_json")

kernel_name="rapids"
echo "--- Install jupyter notebook kernel: ${kernel_name} ---"
conda create -n ${kernel_name} python=3.8.15 -y
source activate ${kernel_name}
conda install -y ipykernel
ipython kernel install --user --name=${kernel_name}
python3 -m pip install -U $package_versions

echo "Running: inference.py $@"
python inference.py $@