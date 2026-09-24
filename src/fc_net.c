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


typedef enum Activaton {
    ACT_NONE,
    ACT_RELU
} Activaton;

typedef struct Layer {
    Matrix W, dW; // (out, dim)
    Vector b, db; // (out, )
    Activaton Act;


    Vector z; // pre activation
    Vector a; // post activation
    Vector Grad; // propagating gradient (out, )
    Vector dinput; // grad of the loss w.r.t this layer's input (in, )
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


void InitVector(MemoryArena *a, Vector *v, int length) {
    v->Data = ArenaPush(a, sizeof(f64) * length);
    if (v->Data == NULL) { printf("failed to allocate for v!\n"); exit(1); }
    v->Length = length;
    memset(v->Data, 0, sizeof(f64) * length);
}
Vector* NewVector(MemoryArena *a, int length) {
    Vector* v = ArenaPush(a, sizeof(Vector));
    InitVector(a, v, length);
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

    int in_dim = cfg->InputDim;
    for (int i = 0; i < cfg->LayerCount; i ++) {
        int is_last = i == cfg->LayerCount - 1;
        int out_dim = is_last ? cfg->OutputDim : cfg->HiddenDim;
        f64 init_scale =  cfg->IsKaimingInit ? sqrt(6.0 / (f64)in_dim) : 0.01;
        Activaton act = is_last ? ACT_NONE : ACT_RELU;
        model->Layers[i].Act = act;
        InitMatrix(a, &model->Layers[i].W, out_dim, in_dim, init_scale);
        InitMatrix(a, &model->Layers[i].dW, out_dim, in_dim, 0.0);
        InitVector(a, &model->Layers[i].b, out_dim);
        InitVector(a, &model->Layers[i].db, out_dim);

        InitVector(a, &model->Layers[i].z, out_dim);
        InitVector(a, &model->Layers[i].a, out_dim);
        InitVector(a, &model->Layers[i].Grad, out_dim);
        InitVector(a, &model->Layers[i].dinput, in_dim);

        in_dim = out_dim; // lord help me
    }
    model->LayerCount = cfg->LayerCount;
    return model;
}

void Forward(Model *m, Vector *x) {
    Vector* current_x = x;

    for (int i = 0; i < m->LayerCount; i ++) {
        f64 *xdata = current_x->Data;
        Layer* L = &m->Layers[i];
        Matrix* W = &L->W;
        Vector* b = &L->b;
        assert(W->ColumnCount == current_x->Length);
        for (int r = 0; r < W->RowCount; r ++) {
            f64 dot = 0.0;
            for (int c = 0; c < W->ColumnCount; c++) {
                dot += W->Data[r * W->ColumnCount + c] * xdata[c];
            }
            L->z.Data[r] = dot + b->Data[r];
        }
        if (m->Layers[i].Act == ACT_RELU) {
            ReLU(&L->z, &L->a);
        } else {
            memcpy(L->a.Data, L->z.Data, sizeof(f64) * L->z.Length);
        }
        current_x = &L->a;
    }
}

void Softmax(Vector *logits, Vector *probs) {
    f64 max_logit = logits->Data[0];
    // max for numerical stability
    // exp(logit - max)
    // divide by sum
    for (int i = 1; i < logits->Length; i++) {
        if (logits->Data[i] > max_logit) max_logit = logits->Data[i];
    }

    f64 sum = 0.0;
    for (int i = 0; i < logits->Length; i++) {
        probs->Data[i] = exp(logits->Data[i] - max_logit);
        sum += probs->Data[i];
    }
    for (int i = 0; i < logits->Length; i++) {
        probs->Data[i] /= sum;
    }
}

void Backward(Model *m, Vector *x, int y) {

    // x => W1(x) + b1 -> z1 => ReLU(z1) -> a1 => W2(a1) + b2 -> logits => softmax(logits) -> probs

    int n = m->LayerCount;

    // start at last ("head") layer
    Layer *last = &m->Layers[n - 1];
    Vector *logits = &last->a;

    // softmax backprop
    // first compute probs
    Softmax(logits, &last->Grad);

    // if we incrase prob for the right label, the loss goes down
    // meaning the grad there is negative
    // which is what we want for the correct label
    // for all other labels keepign the grads positive is signaling
    // that if the model increases those probs the loss will go up 
    // which we want (positive grad will push logits down in DG, and negative grad up)
    // so this pushes the logits towards their indented probablity, y -> 1.0, rest -> 0.0
    last->Grad.Data[y] -= 1.0;

    // layers from the back
    for (int i = n - 1; i >= 0; i--) {
        Layer  *L     = &m->Layers[i];
        Vector *input = (i == 0) ? x : &m->Layers[i - 1].a;

        // activation
        if (L->Act == ACT_RELU) {
            for (int k = 0; k < L->z.Length; k++) {
                L->Grad.Data[k] = L->z.Data[k] > 0.0 ? L->Grad.Data[k] : 0.0;
            }
        }

        // dL/dW, gradient wrt the weights
        // (how does the loss change if we chang the weights)
        for (int r = 0; r < L->W.RowCount; r++) {
            L->db.Data[r] = L->Grad.Data[r];
            f64 *dWrow = &L->dW.Data[r * L->W.ColumnCount];
            for (int c = 0; c < L->W.ColumnCount; c++) {
                dWrow[c] = L->Grad.Data[r] * input->Data[c];
            }
        }

        // dL/dinput, gradient wrt input
        // (how does the loss change if we change the input)
        // need them to chain grad upwards to previous layer
        for (int c = 0; c < L->W.ColumnCount; c++) {
            // dot product is summing products,
            // derivative of a sum is a sum of derivatives (over the row)
            f64 sum = 0.0;
            for (int r = 0; r < L->W.RowCount; r++) {
                sum += L->W.Data[r * L->W.ColumnCount + c] * L->Grad.Data[r];
            }
            L->dinput.Data[c] = sum;
        }

        if (i > 0) {
            Layer *prev = &m->Layers[i - 1];
            assert(L->dinput.Length == prev->Grad.Length);
            // input of this layer is output of prev layer
            // flow grad backward, next iteration treads Grad.Data as grad of activations
            memcpy(prev->Grad.Data, L->dinput.Data, sizeof(f64) * L->dinput.Length);
        }
    }
}



int main(int argc, char *argv[]) {

    srand((u32)1234);

    MemoryArena main_arena = CreateArena((size_t)(10 << 20));
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
    int train_steps = TRAIN_BATCH_COUNT * ENTRIES_PER_BATCH * 2;
    f64 lr[3] = {0.005, 0.001, 0.0001};
    int RUN_TEST = 1;
    int KAIMING_INIT = 1;
    f64 reg = 0.0004;

    // model
    const int modeldim = IMG_SIZE;
    const int hiddendim = 24; 
    const int outdim = 10;
    f64 expected_loss = -log(1.0 / (f64)outdim);

    ModelCreateConfig cfg = {
        .InputDim = IMG_SIZE,
        .HiddenDim = hiddendim,
        .OutputDim = outdim,
        .IsKaimingInit = KAIMING_INIT,
        .LayerCount = 2,
    };
    Model* model = CreateModel(&main_arena, &cfg);

    Vector* x = NewVector(&main_arena, modeldim);
    Vector* probs = NewVector(&main_arena, outdim);

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

        Forward(model, x);

        if (step < 10 || step % 1000 == 0 || step == train_steps - 1) {
            Layer *last = &model->Layers[model->LayerCount - 1];
            Softmax(&last->a, probs);

            f64 reg_loss = 0.0;
            for (int i = 0; i < model->LayerCount; i++) {
                Layer *L = &model->Layers[i];
                int n = L->W.RowCount * L->W.ColumnCount;
                for (int k = 0; k < n; k++)
                    reg_loss += L->W.Data[k] * L->W.Data[k];
            }
            reg_loss *= reg * 0.5;

            f64 loss = -log(probs->Data[y]) + reg_loss;
            printf("step %d loss: %.3f (reg_l: %.3f), expected init loss: %.3f\n",
                step, loss, reg_loss, expected_loss);
        }
    
    
        // backward
        Backward(model, x, y);

        // update weights, sgd
        int lr_idx = step >= (int)((f64)train_steps * 0.9) 
            ? 2 : step >= train_steps / 2 
            ? 1 : 0;

        for (int i = 0; i < model->LayerCount; i++) {
            Layer *L = &model->Layers[i];

            // L2 grad
            for (int k = 0; k < L->W.RowCount * L->W.ColumnCount; k++) {
                L->dW.Data[k] += reg * L->W.Data[k];
            }
            // update w
            for (int k = 0; k < L->W.RowCount * L->W.ColumnCount; k++) {
                L->W.Data[k] -= lr[lr_idx] * L->dW.Data[k];
            }
            for (int k = 0; k < L->b.Length; k++) {
                L->b.Data[k] -= lr[lr_idx] * L->db.Data[k];
            }
        }
    }

    LogArena(&main_arena);

    if (RUN_TEST == 0) {
        return 0;
    }
    // RUN_TEST
    printf("Running test...\n");
    CifarBatch test_batch = LoadCifarBatch("data/cifar-10-batches-bin/test_batch.bin");
    Layer *last = &model->Layers[model->LayerCount - 1];

    f64 loss_accum = 0.0;
    int correct_predictions = 0;
    for (int i = 0; i < ENTRIES_PER_BATCH; i++) {
        CifarEntryView entry_view = GetEntryView(test_batch, i % ENTRIES_PER_BATCH);
        int y = entry_view.Label;
        for (int j = 0; j < modeldim; j++) {
            x->Data[j] = (entry_view.ImageData[j] - 127.5) / 127.5;
        }

        Forward(model, x);
        Softmax(&last->a, probs);

        loss_accum += -log(probs->Data[y]);

        int predicted_y = 0;
        for (int k = 1; k < outdim; k++)
            if (probs->Data[k] > probs->Data[predicted_y]) predicted_y = k;
        if (predicted_y == y) correct_predictions++;
    }

    printf("model: layers: %d h: %d t_steps: %d\n", model->LayerCount, hiddendim, train_steps);
    printf("Test Accuracy: %.2f%% loss avg: %f\n", 
        100.0 * (f64)correct_predictions / (f64)ENTRIES_PER_BATCH,
        loss_accum / (f64)ENTRIES_PER_BATCH);


    
    return 0;
}