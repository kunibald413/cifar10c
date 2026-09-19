

double lr = 0.001;
const int train_steps = ENTRIES_PER_BATCH * 5 * 1; // one epoch
with mean centering achieves:
Test Accuracy: 37.010% loss avg: 1.830146



get binaries: https://cave.cs.toronto.edu/kriz/cifar.html

extract cifar10 batch binaries into data folder so that 
first batch is here "data/cifar-10-batches-bin/test_batch.bin"

compile:

cc src/main.c -o train -O0 -g

cc src/main.c -o train -O3

cc src/main.c -o train -O3 -march=native -funroll-loops


or with cmake:

./run.sh