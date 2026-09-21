#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "cifar10.c"

// compile: cc src/manual_backward.c -o train -O3 -march=native -funroll-loops

typedef uint8_t byte;
typedef uint32_t u32;
typedef float f32;
typedef double f64;

typedef struct Matrix {
    int RowCount, ColumnCount;
    f64* Data; // row major, flat array
} Matrix;

typedef struct Vector {
    int Length;
    f64* Data;
} Vector;

f64 RndF64() {
    return rand() / (f64)RAND_MAX;
}

Vector NewVector(int length) {
    Vector v;
    v.Data = calloc(length, sizeof(f64));
    if (v.Data == NULL) { printf("failed to allocate for v!\n"); exit(1); }
    v.Length = length;
    return v;
}

void ModelForward(Matrix *W, Vector *b, Vector *x, 
    Vector *xout, Vector *xout_exp, Vector *probs) {
    // W @ x + b
    for (int r = 0; r < W->RowCount; r ++) {
        f64 dot = 0.0;
        for (int c = 0; c < W->ColumnCount; c++) {
            dot += W->Data[r * W->ColumnCount + c] * x->Data[c];
        }
        xout->Data[r] = dot + b->Data[r];
    }

    // for (int i = 0; i < xout->Length; i ++) { printf("xout value %f\n", xout->Data[i]); }

    // softmax
    f64 exp_sum = 0.0;
    f64 max_logit = xout->Data[0];
    for (int i = 0; i < xout->Length; i ++) {
        if (xout->Data[i] > max_logit) max_logit = xout->Data[i];
    }

    for (int i = 0; i < xout_exp->Length; i ++) {
        xout_exp->Data[i] = exp(xout->Data[i] - max_logit);
        exp_sum += xout_exp->Data[i];
    }

    //for (int i = 0; i < xout_exp->Length; i ++) { printf("xout_exp->value %f\n", xout_exp->Data[i]); }
    //printf("exp_sum: %f\n", exp_sum);
    
    for (int i = 0; i < xout_exp->Length; i ++) {
        probs->Data[i] = xout_exp->Data[i] / exp_sum;
        // printf("probs value %f\n", probs.Data[i]);
    }
    
}

void Backward(Matrix *W, Vector *b, Vector *x, Vector *xout, Vector *probs, int y, 
    Matrix* dW, Vector *db, Vector* dz) {
    // fused softmax and crossentropy
    // full graph (exp -> sum -> div -> log) is a pile of shit 
    // but every intermediate derivative cancels to to just: 
    //    dL/dz_i = p_i - (i == y ? 1 : 0)
    for (int i = 0; i < dz->Length; i++) {
        dz->Data[i] = probs->Data[i];
    }
    dz->Data[y] -= 1.0;

    for (int r = 0; r < W->RowCount; r++) {
        // bias grad is just 1, so simplified from dz.Data[r] * 1
        db->Data[r] = dz->Data[r]; 
        
        for (int c = 0; c < W->ColumnCount; c++) { 
            // in forward weight param just mutliplies elementwise with input (x)
            // local grad for weight param is then just the x elem
            dW->Data[r * W->ColumnCount + c] = dz->Data[r] * x->Data[c];
        }
    }
}

int main(int argc, char *argv[]) {

    srand((u32)1234);

    // load data
    #define TRAIN_BATCH_COUNT 5
    CifarBatch train_batches[TRAIN_BATCH_COUNT];
    for (int i = 0; i < TRAIN_BATCH_COUNT; i ++) {
        char batch_path[64];
        snprintf(batch_path, sizeof(batch_path), "data/cifar-10-batches-bin/data_batch_%d.bin", i + 1);
        printf("batch_path: %s\n", batch_path);
        train_batches[i] = LoadCifarBatch(batch_path);
        if (train_batches[i].EntryCount < ENTRIES_PER_BATCH || train_batches[i].RawData == NULL) {
            printf("Failed to load batch %s! entrycount: %d\n Aborting.",  batch_path, train_batches[i].EntryCount);
            exit(1);
        }
    }

    // config
    int train_steps = TRAIN_BATCH_COUNT * ENTRIES_PER_BATCH * 2;
    f64 lr[3] = {0.001, 0.0001, 0.00001};
    int RUN_TEST = 1;
    // Test Accuracy: 41.300% loss avg: 1.708759


    // model
    const int modeldim = IMG_SIZE;
    const int outdim = 10;

    f64 expected_loss = -log(1.0 / (f64)outdim);

    Matrix W;
    W.Data = calloc(modeldim * outdim, sizeof(f64));
    if (W.Data == NULL) {
        printf("failed to allocate matrix!\n");
        exit(1);
    }
    W.RowCount = outdim;
    W.ColumnCount = modeldim;
    for (int i = 0; i < W.RowCount * W.ColumnCount; i ++) {
        W.Data[i] = (RndF64() * 2 - 1) * 0.01;
        //printf("inited W to %f\n", W.Data[i]);
    }
    
    Vector b = NewVector(W.RowCount);
    for (int i = 0; i < b.Length; i ++) {
        b.Data[i] = 0.0;
        //printf("inted bias to %f\n", b.Data[i]);
    }

    // grad buffers
    Matrix dW;
    dW.Data = calloc(W.RowCount * W.ColumnCount, sizeof(f64));
    if (dW.Data == NULL) { printf("alloc fail for dW\n"); exit(1); }
    dW.RowCount = W.RowCount;
    dW.ColumnCount = W.ColumnCount;
    Vector db = NewVector(b.Length);
    Vector dz = NewVector(W.RowCount);

    // buffers
    Vector x = NewVector(modeldim);
    Vector xout = NewVector(W.RowCount);
    assert(x.Length == W.ColumnCount && "x W shape mismatch");
    assert(b.Length == W.RowCount && "b W shape mismatch");
    assert(xout.Length == W.RowCount && "xout W shape mismatch");

    Vector xout_exp = NewVector(xout.Length);
    Vector probs = NewVector(xout.Length);

    
    for (int step = 0; step < train_steps; step ++) {

        int batch_idx = (step / ENTRIES_PER_BATCH) % TRAIN_BATCH_COUNT;
        int entry_idx = step % ENTRIES_PER_BATCH;
        CifarEntryView x_entry = GetEntryView(train_batches[batch_idx], entry_idx);

        int y = x_entry.Label;
        for (int i = 0; i < modeldim; i ++) {
            //x.Data[i] = 0.001; // Dummy input
            x.Data[i] = (x_entry.ImageData[i] - 127.5) / 127.5;
            // printf("inted x to %f\n", x.Data[i]);
        }
        
        ModelForward(&W, &b, &x, &xout, &xout_exp, &probs);
        
        if (step < 50 || step % 1000 == 0 || step == train_steps - 1) {
            f64 loss = -log(probs.Data[y]);
            printf("step %d loss: %.4f, expected init loss: %.4f \n", step, loss, expected_loss);
        }
    
    
        // backward
    
        // zero grads
        memset(dW.Data, 0, W.RowCount * W.ColumnCount * sizeof(f64));
        memset(db.Data, 0, b.Length * sizeof(f64));
    
        Backward(&W, &b, &x, &xout, &probs, y, &dW, &db, &dz);
    
        // update weights, sgd
        int lr_idx = step >= (int)((f64)train_steps * 0.9) 
            ? 2 : step >= train_steps / 2 
            ? 1 : 0;
        for (int i = 0; i < W.RowCount * W.ColumnCount; i++) {
            W.Data[i] -= lr[lr_idx] * dW.Data[i];
        }
        for (int i = 0; i < b.Length; i++) {
            b.Data[i] -= lr[lr_idx] * db.Data[i];
        }

    }


    if (RUN_TEST == 0) {
        return 0;
    }
    // RUN_TEST
    printf("Running test...\n");
    CifarBatch test_batch = LoadCifarBatch("data/cifar-10-batches-bin/test_batch.bin");
    f64 loss_accum = 0.0;
    int correct_predictions = 0;
    for (int i = 0; i < ENTRIES_PER_BATCH; i ++) {
        CifarEntryView entry_view = GetEntryView(test_batch, i % ENTRIES_PER_BATCH);
        int y = entry_view.Label;
        for (int i = 0; i < modeldim; i ++) {
            x.Data[i] = (entry_view.ImageData[i] - 127.5) / 127.5;
        }
        ModelForward(&W, &b, &x, &xout, &xout_exp, &probs);

        f64 loss = -log(probs.Data[y]);
        loss_accum += loss;

        int predicted_y = 0;
        for (int i = 1; i < outdim; i++) {
            if (probs.Data[i] > probs.Data[predicted_y]) predicted_y = i;
        }
        if (predicted_y == y) correct_predictions++;

        // printf("loss: %.4f, \n", loss, expected_loss);
    }

    printf("Test Accuracy: %.3f%% loss avg: %f\n", 
        100.0 * (f64)correct_predictions / (f64)ENTRIES_PER_BATCH,
        loss_accum / (f64)ENTRIES_PER_BATCH);


    
    return 0;
}