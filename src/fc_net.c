#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "cifar10.c"

// compile: cc src/fc_net.c -o train -O3 -march=native -funroll-loops

// fully connected net, with ReLU nonlinearity
// softmaxloss
// D -> input dimension
// H -> hidden dimension
// C -> amout of classes

// Affine: y = W @ x + b  (aka "Linear" in torch, math term is `affine` since b shifts origin)

// Affine -> ReLU -> Affine -> Softmax

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


typedef struct Layer {
    Matrix W; // (out, dim)
    Vector b; // (out, )
} Layer;

typedef struct Model {
    Layer* Layers;
    int LayerCount;
} Model;

typedef struct ModelCreateConfig {
    int InputDim;
    int HiddenDim;
    int OutputDim;
    int LayerCount;
    int IsKaimingInit;
} ModelCreateConfig;

#define ARENA_ALIGN ((size_t)16)
typedef struct MemoryArena {
    byte* Base;
    size_t Used;
    size_t Capacity;
} MemoryArena;

MemoryArena CreateArena(size_t capacity) {
    MemoryArena a;

    void* ptr = calloc(capacity, sizeof(byte));
    if (ptr == NULL) { 
        fprintf(stderr, "failed to allocate arena bytes %zu\n", capacity); 
        abort();
    }
    a.Base = (byte*)ptr;
    a.Used = 0;
    a.Capacity = capacity;

    return a;
}

size_t GetArenaAlignOffset(MemoryArena *a, size_t alignment) {
    size_t current_address = (size_t)(a->Base + a->Used);
    size_t mask = alignment - 1;
    if (current_address & mask) { // fast way to get remainder
        return alignment - (current_address & mask);
    }
    return 0;
}

void* ArenaPushAligned(MemoryArena *a, size_t size, size_t alignment) {
    size_t offset = GetArenaAlignOffset(a, alignment);
    size_t total = size + offset;

    if (total > a->Capacity - a->Used) {
        fprintf(stderr,
                "Arena out of capacity: requested %zu (with %zu align padding), "
                "capacity %zu, used %zu, remaining %zu\n",
                size, offset, a->Capacity, a->Used, a->Capacity - a->Used);
        abort();
    }

    void* result = a->Base + a->Used + offset;
    a->Used += total;
    return result;
}

void* ArenaPush(MemoryArena *a, size_t size) {
    return ArenaPushAligned(a, size, ARENA_ALIGN);
}

void LogArena(MemoryArena *a) {
    printf("arena used: %zu cap: %zu (%.3f%%) current ptr: %p\n", 
        a->Used, a->Capacity, 
        (f64)a->Used / (f64)a->Capacity * 100.0,
        a->Base + a->Used);
}

f64 RndF64() {
    return rand() / (f64)RAND_MAX;
}



void InitMatrix(MemoryArena *a, Matrix *wptr, int rows, int columns, f64 init_scale) {
    wptr->Data = (f64*)ArenaPush(a, rows * columns * sizeof(f64));
    if (wptr->Data == NULL) {
        printf("failed to allocate matrix!\n");
        exit(1);
    }
    wptr->RowCount = rows;
    wptr->ColumnCount = columns;

    if (init_scale > 0.0) {
        for (int i = 0; i < wptr->RowCount * wptr->ColumnCount; i ++) {
            wptr->Data[i] = (RndF64() * 2 - 1) * init_scale;
        }
    }
}

Matrix* CreateMatrix(MemoryArena *a, int rows, int columns, f64 random_init_scale) {
    Matrix *wptr = (Matrix*)ArenaPush(a, sizeof(Matrix));
    InitMatrix(a, wptr, rows, columns, random_init_scale);
    return wptr;
}


void InitVector(Vector *v, int length) {
    v->Data = calloc(length, sizeof(f64));
    if (v->Data == NULL) { printf("failed to allocate for v!\n"); exit(1); }
    v->Length = length;
}
Vector* NewVector(MemoryArena *a, int length) {
    Vector* v = ArenaPush(a, sizeof(Vector));
    InitVector(v, length);
    return v;
}


void ReLU(Vector *x, Vector *xout) {
    assert(x->Length == xout->Length && "ReLU, vector length mismatch");
    for (int i = 0; i < x->Length; i ++) {
        xout->Data[i] = x->Data[i] > 0.0 ? x->Data[i] : 0.0;
    }
}

Model* CreateModel(MemoryArena *a, ModelCreateConfig * cfg) {
    assert(cfg->LayerCount >= 1 && "Model must have at least one layer");
    Model* model = (Model*)ArenaPush(a, sizeof(Model));
    model->Layers = ArenaPush(a, sizeof(Layer) * cfg->LayerCount);

    int indim = cfg->InputDim;
    for (int i = 0; i < cfg->LayerCount; i ++) {
        int out = i == cfg->LayerCount - 1 ? cfg->OutputDim : cfg->HiddenDim;
        f64 init_scale =  cfg->IsKaimingInit ? sqrt(2.0 / (f64)indim) : 0.01;
        InitMatrix(a, &model->Layers[i].W, out, indim, init_scale);
    }
    return model;
}


void ModelForward(Matrix *W, Vector *b, 
    Matrix *W2, Vector *b2,
    Vector *x, 
    Vector *xout, Vector *xout_relu, 
    Vector *xout2, 
    Vector *xout_exp, Vector *probs
) {
    
    assert(x->Length == W->ColumnCount && "W.col must mach x.length");
    assert(b->Length == W->RowCount && "W.row must mach b.length");
    assert(xout->Length == W->RowCount && "W.row must mach xout.length");

    assert(xout_relu->Length == xout->Length     && "relu out must match relu in");

    assert(W2->ColumnCount == xout_relu->Length && "W2.cols must match relu.len");
    assert(b2->Length == W2->RowCount && "b2.len must match W2.rows");
    assert(xout2->Length == W2->RowCount && "xout2.len must match W2.rows");

    assert(xout2->Length == xout_exp->Length && "xout2 and xout_exp vector length mismatch");
    assert(xout2->Length == probs->Length && "xout2 and probs vector length mismatch");

    // W1 @ x + b
    // out: (num_classes,)
    for (int r = 0; r < W->RowCount; r ++) {
        f64 dot = 0.0;
        for (int c = 0; c < W->ColumnCount; c++) {
            dot += W->Data[r * W->ColumnCount + c] * x->Data[c];
        }
        xout->Data[r] = dot + b->Data[r];
    }

    // for (int i = 0; i < xout->Length; i ++) { printf("xout value %f\n", xout->Data[i]); }
    
    // relu
    ReLU(xout, xout_relu);

    // W2 @ x + b2
    for (int r = 0; r < W2->RowCount; r ++) {
        f64 dot = 0.0;
        for (int c = 0; c < W2->ColumnCount; c++) {
            dot += W2->Data[r * W2->ColumnCount + c] * xout_relu->Data[c];
        }
        xout2->Data[r] = dot + b2->Data[r];
    }

    // softmax
    f64 exp_sum = 0.0;
    f64 max_logit = xout2->Data[0];
    for (int i = 0; i < xout2->Length; i ++) {
        if (xout2->Data[i] > max_logit) max_logit = xout2->Data[i];
    }

    // (num_classes,)
    for (int i = 0; i < xout_exp->Length; i ++) {
        xout_exp->Data[i] = exp(xout2->Data[i] - max_logit);
        exp_sum += xout_exp->Data[i];
    }

    //for (int i = 0; i < xout_exp->Length; i ++) { printf("xout_exp->value %f\n", xout_exp->Data[i]); }
    //printf("exp_sum: %f\n", exp_sum);
    
    // (num_classes,)
    for (int i = 0; i < xout_exp->Length; i ++) {
        probs->Data[i] = xout_exp->Data[i] / exp_sum;
        // printf("probs value %f\n", probs.Data[i]);
    }
    
}

void Backward(
    // forward buffers
    Matrix *W, Vector *b, Matrix *W2, Vector *b2,
    Vector *x, Vector *xout, Vector *xout_relu, Vector *xout2, Vector *probs,
    int y,
    // grad buffers
    Matrix *dW, Vector *db, Matrix *dW2, Vector *db2,
    Vector *dxout2, Vector *dxout_relu
) {
    // fused softmax and crossentropy
    // full graph (exp -> sum -> div -> log) is a pile of shit 
    // but every intermediate derivative cancels to to just: 
    //    dL/dz_i = p_i - (i == y ? 1 : 0)
    for (int i = 0; i < dxout2->Length; i++) {
        dxout2->Data[i] = probs->Data[i];
    }
    dxout2->Data[y] -= 1.0;

    // W2
    // xout2 = W2 @ xout_relu + b2
    for (int r = 0; r < W2->RowCount; r++) {
        db2->Data[r] = dxout2->Data[r];
        for (int c = 0; c < W2->ColumnCount; c++) {
            dW2->Data[r * W2->ColumnCount + c] = dxout2->Data[r] * xout_relu->Data[c];
        }
    }

    for (int c = 0; c < W2->ColumnCount; c++) {
        f64 sum = 0.0;
        for (int r = 0; r < W2->RowCount; r++) {
            sum += W2->Data[r * W2->ColumnCount + c] * dxout2->Data[r];
        }
        dxout_relu->Data[c] = sum;
    }

    // relu
    // max(0, x) means no gradiesnt when below 0
    for (int i = 0; i < dxout_relu->Length; i++) {
        dxout_relu->Data[i] = xout->Data[i] > 0.0 ? dxout_relu->Data[i] : 0.0;
    }


    // W1
    for (int r = 0; r < W->RowCount; r++) {
        // bias grad is just 1, so simplified from dxout_relu.Data[r] * 1
        db->Data[r] = dxout_relu->Data[r]; 
        
        for (int c = 0; c < W->ColumnCount; c++) { 
            // in forward weight param just mutliplies elementwise with input (x)
            // local grad for weight param is then just the x elem
            dW->Data[r * W->ColumnCount + c] = dxout_relu->Data[r] * x->Data[c];
        }
    }
}


int main(int argc, char *argv[]) {

    srand((u32)1234);

    MemoryArena main_arena = CreateArena((size_t)(30 << 20));
    MemoryArena scratch_arena = CreateArena((size_t)(1 << 20));

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
    int train_steps = TRAIN_BATCH_COUNT * ENTRIES_PER_BATCH * 1;
    f64 lr[3] = {0.005, 0.001, 0.0001};
    int RUN_TEST = 1;
    int KAIMING_INIT = 1;
    f64 reg = 0.0004;

    train_steps = 1000;

    // model
    const int modeldim = IMG_SIZE;
    const int hiddendim = 32; 
    const int outdim = 10;
    f64 expected_loss = -log(1.0 / (f64)outdim);

    ModelCreateConfig cfg = {
        .InputDim = IMG_SIZE,
        .HiddenDim = 32,
        .OutputDim = outdim,
        .IsKaimingInit = KAIMING_INIT,
        .LayerCount = 1,
    };
    Model* model = CreateModel(&main_arena, &cfg);



    f64 init_scale1 = KAIMING_INIT ? sqrt(2.0 / (f64)modeldim) : 0.01;
    Matrix* W1 = CreateMatrix(&main_arena, hiddendim, modeldim, init_scale1);
    Vector* b1 = NewVector(&main_arena, W1->RowCount);
    for (int i = 0; i < b1->Length; i ++) { b1->Data[i] = 0.0; }

    f64 init_scale2 = KAIMING_INIT ? sqrt(2.0 / (f64)hiddendim) : 0.01;
    Matrix* w_out = CreateMatrix(&main_arena, outdim, hiddendim, init_scale2);
    Vector* b_out = NewVector(&main_arena, w_out->RowCount);
    for (int i = 0; i < b_out->Length; i ++) { b_out->Data[i] = 0.0; }

    // grad buffers
    Matrix* dW = CreateMatrix(&main_arena, W1->RowCount, W1->ColumnCount, 0.0);
    Vector* db = NewVector(&main_arena, b1->Length);
    Vector* dxout_relu = NewVector(&main_arena, W1->RowCount);
    
    Matrix* dW2 = CreateMatrix(&main_arena, w_out->RowCount, w_out->ColumnCount, 0.0);
    Vector* db2 = NewVector(&main_arena, b_out->Length);
    Vector* dxout2 = NewVector(&main_arena, w_out->RowCount);

    // buffers
    Vector* x = NewVector(&main_arena, modeldim);
    Vector* xout = NewVector(&main_arena, W1->RowCount);
    Vector* xout_relu = NewVector(&main_arena, xout->Length);
    Vector* xout2 = NewVector(&main_arena, w_out->RowCount);
    assert(x->Length == W1->ColumnCount && "x W shape mismatch");
    assert(b1->Length == W1->RowCount && "b W shape mismatch");
    assert(xout->Length == W1->RowCount && "xout W shape mismatch");
    assert(xout2->Length == w_out->RowCount && "xout W shape mismatch");

    Vector* xout_exp = NewVector(&main_arena, xout2->Length);
    Vector* probs = NewVector(&main_arena, xout2->Length);

    LogArena(&main_arena);

    
    for (int step = 0; step < train_steps; step ++) {

        int batch_idx = (step / ENTRIES_PER_BATCH) % TRAIN_BATCH_COUNT;
        int entry_idx = step % ENTRIES_PER_BATCH;
        CifarEntryView x_entry = GetEntryView(train_batches[batch_idx], entry_idx);

        int y = x_entry.Label;
        for (int i = 0; i < modeldim; i ++) {
            //x.Data[i] = 0.001; // Dummy input
            x->Data[i] = (x_entry.ImageData[i] - 127.5) / 127.5;
            // printf("inted x to %f\n", x.Data[i]);
        }

        ModelForward(W1, b1, w_out, b_out, x, xout, xout_relu, xout2, xout_exp, probs);

        f64 reg_loss = 0.0;
        for (int i = 0; i < W1->RowCount * W1->ColumnCount; i ++) reg_loss += W1->Data[i] * W1->Data[i];
        for (int i = 0; i < w_out->RowCount * w_out->ColumnCount; i ++) reg_loss += w_out->Data[i] * w_out->Data[i];
        reg_loss = reg_loss * reg * 0.5; // 0.5 so that grad = reg*W instead of 2*reg*W
        
        if (step < 10 || step % 1000 == 0 || step == train_steps - 1) {
            f64 loss = -log(probs->Data[y]) + reg_loss;
            printf("step %d loss: %.3f (reg_l: %.3f), expected init loss: %.3f \n", 
                step, loss, reg_loss, expected_loss);
        }
    
    
        // backward
    
        // zero grads
        memset(dW->Data, 0, W1->RowCount * W1->ColumnCount * sizeof(f64));
        memset(db->Data, 0, b1->Length * sizeof(f64));
        memset(dW2->Data, 0, w_out->RowCount * w_out->ColumnCount * sizeof(f64));
        memset(db2->Data, 0, b_out->Length * sizeof(f64));
    
        Backward(W1, b1, w_out, b_out, 
            x, xout, xout_relu, xout2, probs, y, 
            dW, db, dW2, db2, dxout2, dxout_relu
        );

        // apply reg grads
        for (int i = 0; i < W1->RowCount * W1->ColumnCount; i++) {
            dW->Data[i] += reg * W1->Data[i];
        }
        for (int i = 0; i < w_out->RowCount * w_out->ColumnCount; i++) {
            dW2->Data[i] += reg * w_out->Data[i];
        }
            
        // update weights, sgd
        int lr_idx = step >= (int)((f64)train_steps * 0.9) 
            ? 2 : step >= train_steps / 2 
            ? 1 : 0;

        for (int i = 0; i < W1->RowCount * W1->ColumnCount; i++) {
            W1->Data[i] -= lr[lr_idx] * dW->Data[i];
        }
        for (int i = 0; i < b1->Length; i++) {
            b1->Data[i] -= lr[lr_idx] * db->Data[i];
        }

        for (int i = 0; i < w_out->RowCount * w_out->ColumnCount; i++) {
            w_out->Data[i] -= lr[lr_idx] * dW2->Data[i];
        }
        for (int i = 0; i < b_out->Length; i++) {
            b_out->Data[i] -= lr[lr_idx] * db2->Data[i];
        }

    }

    LogArena(&main_arena);

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
            x->Data[i] = (entry_view.ImageData[i] - 127.5) / 127.5;
        }
        ModelForward(W1, b1, w_out, b_out, x, xout, xout_relu, xout2, xout_exp, probs);

        f64 loss = -log(probs->Data[y]);
        loss_accum += loss;

        int predicted_y = 0;
        for (int i = 1; i < outdim; i++) {
            if (probs->Data[i] > probs->Data[predicted_y]) predicted_y = i;
        }
        if (predicted_y == y) correct_predictions++;

        // printf("loss: %.4f, \n", loss, expected_loss);
    }

    printf("Test Accuracy: %.2f%% loss avg: %f\n", 
        100.0 * (f64)correct_predictions / (f64)ENTRIES_PER_BATCH,
        loss_accum / (f64)ENTRIES_PER_BATCH);


    
    return 0;
}