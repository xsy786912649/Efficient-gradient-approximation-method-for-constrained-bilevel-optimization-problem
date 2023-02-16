Dependencies
Python 3.7+
pytorch 1.7.0+
qpth 0.0.11+ （not required)
tqdm
cvxpy 1.2.0
torchnet

Training:
Run: python train.py --gpu 0 --save-path "./experiments/MetaOptNet_SVM" --train-shot 15 --head SVM --network ResNet --dataset CIFAR_FS --eps 0.1

Test:
python test.py --gpu 0 --load ./experiments/MetaOptNet_SVM --episode 1000 --way 5 --shot 5 --query 15 --head SVM --network ResNet --dataset CIFAR_FS

Dataset download link:
https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view
https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view

Note: modify the address of dataset in data/CIFAR_FS.py and data/FC100.py