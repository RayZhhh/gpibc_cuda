Remove -Item res.csv
nvcc -o run main.cu

./run jaffe 128 128 250
./run cifar 32 32 20
./run kth 128 128 20
./run coil 128 128 125
./run mnist 28 28 20
./run uiuc 40 100 70