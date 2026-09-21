#include "cifar10.c"

typedef float    f32;
typedef double   f64;

f64 RandomF64() {
    return (f64)rand() / (f64)RAND_MAX;
}

#define ARENA_ALIGN ((size_t)16)

typedef struct MemoryArena {
    byte* Base;
    size_t Used;
    size_t Capacity;
} MemoryArena;

size_t GetArenaAlignmentOffset(MemoryArena *a, size_t alignment) {
    size_t address = (size_t)(a->Base + a->Used);
    size_t mask = alignment - 1;
    if (address & mask) {
        return alignment - (address & mask);
    }
    return 0;
}

void* MemoryArenaPushAligned(MemoryArena *a, size_t size, size_t alignment) {
    size_t offset = GetArenaAlignmentOffset(a, alignment);
    size_t total = offset + size;
    if (total > a->Capacity - a->Used) {
        return NULL;
    }
    void* result = a->Base + a->Used + offset;
    a->Used += total;
    return result;
}

void* MemoryArenaPush(MemoryArena *a, size_t size) {
    return MemoryArenaPushAligned(a, size, ARENA_ALIGN);
}

MemoryArena MemoryArenaNew(size_t capacity) {
    MemoryArena result;
    void* base = malloc(capacity);
    assert(base != NULL && "Failed to allocate for arena");

    result.Base = (byte *)base;
    result.Used = 0;
    result.Capacity = capacity;
    return result;
}

void LogArena(MemoryArena *a) {
    printf("arena used: %zu cap: %zu (%.3f%%) current ptr: %p\n", 
        a->Used, a->Capacity, 
        (f64)a->Used / (f64)a->Capacity,
        a->Base + a->Used);
}

typedef struct Value Value;
struct Value {
    f64 Data;
    f64 Grad;
    u8 ChildCount;
    Value* Children[2];
    f64 LocalGrads[2];
    f64 _pad; // get to 64 bytes
};

void VAdd(Value* out, Value* a, Value* b) {
    out->Data = a->Data + b->Data;
    out->ChildCount = 2;
    out->Children[0] = a;
    out->Children[1] = b;
    out->LocalGrads[0] = 1.0;
    out->LocalGrads[1] = 1.0;
}
void VMul(Value* out, Value* a, Value* b) {
    out->Data = a->Data * b->Data;
    out->ChildCount = 2;
    out->Children[0] = a;
    out->Children[1] = b;
    out->LocalGrads[0] = b->Data;
    out->LocalGrads[1] = a->Data;
}

void VDiv(Value* out, Value* a, Value* b) {
    f64 bd = b->Data;
    out->Data = a->Data / bd;
    out->ChildCount = 2;
    out->Children[0] = a;
    out->Children[1] = b;
    out->LocalGrads[0] = 1.0 / bd;              
    out->LocalGrads[1] = -a->Data / (bd * bd); 
}

void VNegative(Value* out, Value* x) {
    out->Data = -(x->Data);
    out->ChildCount = 1;
    out->Children[0] = x;
    out->LocalGrads[0] = -1.0;
}


void VLog(Value* out, Value* x) {
    out->Data = log(x->Data); // base e =~ 2.71828
    out->ChildCount = 1;
    out->Children[0] = x;
    out->LocalGrads[0] = 1.0 / x->Data;
}

void VExp(Value* out, Value* x) {
    out->Data = exp(x->Data);
    out->ChildCount = 1;
    out->Children[0] = x;
    out->LocalGrads[0] = exp(x->Data);
}

Value* NewValue(MemoryArena *fwd_a) {
    Value* v = MemoryArenaPush(fwd_a, sizeof(Value));
    v->Grad = 0.0;
    v->ChildCount = 0;
    return v;
}


typedef struct Matrix {
    u32 RowCount, ColumnCount;
    Value* Elements; // flat array, row-major
} Matrix;

Matrix NewMatrix(u32 rows, u32 columns, MemoryArena *a) {
    Matrix mat;
    void* elemptr = MemoryArenaPush(a, (size_t)(rows * columns) * sizeof(Value));
    assert(elemptr != NULL && "Failed ot allocate for Matrix");
    mat.Elements = (Value *)elemptr;
    mat.RowCount = rows;
    mat.ColumnCount = columns;
    return mat;
}

void MatrixRandomInit(Matrix *m) {
    for (int i = 0; i < m->RowCount * m->ColumnCount; i ++) {
        m->Elements[i].Data = (RandomF64() * 2 - 1) * 0.01; // center arount zero
        m->Elements[i].Grad = 0;
        m->Elements[i].ChildCount = 0;
    }
}

void Linear(Matrix* W, Value** out, Value* x, MemoryArena *a) {
    for (int row = 0; row < W->RowCount; row++) {
        Value* sum =  NewValue(a);
        VMul(sum, &W->Elements[row * W->ColumnCount + 0], x + 0);

        // start with col = 1 because first product (col = 0) is computed above
        for (int column = 1; column < W->ColumnCount; column++){
            Value* w = &W->Elements[row * W->ColumnCount + column];
            Value* x_i = x + column;
            Value* product = NewValue(a);
            VMul(product, w, x_i);

            Value* new_sum = NewValue(a);
            VAdd(new_sum, sum, product);

            sum = new_sum;
        }
        out[row] = sum;
    }
}

void SoftMax(Value** x, Value** out, int length, MemoryArena* fwd, MemoryArena *scratch) {
    // exp(elem) / sum ([exp(elem), ...])
    Value** exps = MemoryArenaPush(scratch, sizeof(Value*) * length);
    for (int i = 0; i < length; i++) {
        exps[i] = NewValue(fwd);
        VExp(exps[i], x[i]);
    }

    Value* sum = exps[0];
    for (int i = 1; i < length; i++) {
        Value* new_sum = NewValue(fwd);
        VAdd(new_sum, sum, exps[i]);
        sum = new_sum;
    }

    for (int i = 0; i < length; i++) {
        out[i] = NewValue(fwd);
        VDiv(out[i], exps[i], sum);
    }
}

void Backward(MemoryArena* fwd_a) {
    Value* end = (Value*)(fwd_a->Base + fwd_a->Used) - 1;
    Value* start = (Value*)(fwd_a->Base);

    end->Grad = 1.0;
    // printf("End node value: %f\n", end->Data);

    for (Value* v = end; v >= start; v--) {
        for (int i = 0; i < v->ChildCount; i++) {
            v->Children[i]->Grad += v->Grad * v->LocalGrads[i];
        }
    }
}


int main(int argc, char *argv[]) {

    assert(sizeof(Value) % ARENA_ALIGN == 0 && "sizeof Value must be multiple of 16");


    /* 
        double lr = 0.001;
        int train_steps = ENTRIES_PER_BATCH * 5;
        Test Accuracy: 27.350% loss avg: 2.114724

        double lr = 0.01;
        int train_steps = ENTRIES_PER_BATCH * 5;
        Test Accuracy: 21.520% loss avg: 7.620390

        double lr = 0.002;
        int train_steps = ENTRIES_PER_BATCH * 5
        Test Accuracy: 21.730% loss avg: 2.688580

        double lr = 0.001;
        const int train_steps = ENTRIES_PER_BATCH * 5 * 5;
        Test Accuracy: 28.750% loss avg: 2.111006

        double lr = 0.001;
        const int train_steps = ENTRIES_PER_BATCH * 5 * 1;
        x[i].Data = ((f64)entry_view.ImageData[i] - 127.5) / 127.5; // ~ [-1, 1]
        Test Accuracy: 37.010% loss avg: 1.830146

        double lr = 0.001;
        const int train_steps = ENTRIES_PER_BATCH * 5 * 2;
        x[i].Data = ((f64)entry_view.ImageData[i] - 127.5) / 127.5; // ~ [-1, 1]
        Test Accuracy: 37.070% loss avg: 1.833495

        + bias
        double lr = 0.001;
        const int train_steps = ENTRIES_PER_BATCH * 5 * 1;
        x[i].Data = ((f64)entry_view.ImageData[i] - 127.5) / 127.5; // ~ [-1, 1]
        Test Accuracy: 37.440% loss avg: 1.807230
        
     */

    const int MODEL_DIM = IMG_SIZE;
    const int CIFAR_CLASSES = 10;
    double lr = 0.001;
    const int train_steps = ENTRIES_PER_BATCH * TRAIN_BATCHES * 1;
    const int RUN_TEST = 1;

    srand((u32)1337);

    /// DATA
    const int train_batch_count = 5;
    CifarBatch train_batches[TRAIN_BATCHES];

    for (int i = 0; i < train_batch_count; i ++) {
        char batch_path[64];
        snprintf(batch_path, sizeof(batch_path), "data/cifar-10-batches-bin/data_batch_%d.bin", i + 1);
        printf("batch_path: %s\n", batch_path);
        train_batches[i] = LoadCifarBatch(batch_path);
        if (train_batches[i].EntryCount < ENTRIES_PER_BATCH || train_batches[i].RawData == NULL) {
            printf("Failed to load batch %s! entrycount: %d\n Aborting.",  batch_path, train_batches[i].EntryCount);
            exit(1);
        }
    }

    /// MODEL AND TRAIN
    MemoryArena weights_arena = MemoryArenaNew((size_t)(5 << 20));
    MemoryArena scratch_arena = MemoryArenaNew((size_t)(1 << 20));
    MemoryArena forward_arena = MemoryArenaNew((size_t)(5 << 20));
    LogArena(&weights_arena);
    
    printf("size of Value: %zu\n", sizeof(Value));
    

    Matrix W = NewMatrix(CIFAR_CLASSES, MODEL_DIM, &weights_arena);
    LogArena(&weights_arena);
    Value* bias = MemoryArenaPush(&weights_arena, sizeof(Value) * CIFAR_CLASSES);
    for (int i = 0; i < CIFAR_CLASSES; i++) {
        bias[i].Grad = 0.0;
        bias[i].Data = 0.0;
        bias[i].ChildCount = 0;
    }


    MatrixRandomInit(&W);

    printf("inital grad of W 0,0 %f\n", W.Elements[0].Grad);

    double plausible_init_loss = -log(1.0/(double)CIFAR_CLASSES);
    printf("plausible_init_loss: %f\n", plausible_init_loss);
    

    // TRAIN
    for (int step = 0; step < train_steps; step++) {
        // prepare input
        int batch_idx = (step / ENTRIES_PER_BATCH) % train_batch_count;
        int entry_idx = step % ENTRIES_PER_BATCH;
        CifarEntryView entry_view = GetEntryView(train_batches[batch_idx], entry_idx);

        Value* x = MemoryArenaPush(&forward_arena, sizeof(Value) * IMG_SIZE);
        for (int i = 0; i < IMG_SIZE; i++) {
            //x[i].Data = (f64)entry_view.ImageData[i] / 255.0; // normalize to [0..1]
            x[i].Data = ((f64)entry_view.ImageData[i] - 127.5) / 127.5; // ~ [-1, 1]
            x[i].Grad = 0;
            x[i].ChildCount = 0;
        }
        int y_idx = entry_view.Label;
        
        // forward
        // W @ x + b
        Value** out = MemoryArenaPush(&scratch_arena, sizeof(Value*) * CIFAR_CLASSES);
        Linear(&W, out, x, &forward_arena);
        for (int i = 0; i < CIFAR_CLASSES; i++) {
            Value* elem_with_bias = NewValue(&forward_arena);
            VAdd(elem_with_bias, out[i], &bias[i]);
            out[i] = elem_with_bias;
        }

        
        // calculate softmax and loss for correct label
        Value** probs = MemoryArenaPush(&scratch_arena, sizeof(Value*) * CIFAR_CLASSES);
        SoftMax(out, probs, CIFAR_CLASSES, &forward_arena, &scratch_arena);
        
        Value* y = probs[y_idx];
        Value* log_prob = NewValue(&forward_arena);
        VLog(log_prob, y);

        Value* loss = NewValue(&forward_arena);
        VNegative(loss, log_prob);

        if (train_steps <= 100 || step % 500 == 0) {
            printf("step: %d, loss: %.4f lr: %.5f, b_i: %d e_i: %d\n",
                step+1, loss->Data, lr, batch_idx, entry_idx);
        }
        
        // zero grad
        for (int i = 0; i < W.RowCount * W.ColumnCount; i++)  {
            W.Elements[i].Grad = 0.0;
        }
        for (int i = 0; i < CIFAR_CLASSES; i++) {
            bias[i].Grad = 0.0;
        }

        Backward(&forward_arena);
        
        #if 0
        printf("step grad of W 0,0 %f\n", W.Elements[0].Grad);
        //printf("step grad of E 0,0 %f\n", E.Elements[token * E.ColumnCount].Grad);
        #endif

        // update weights
        for (int i = 0 ; i < W.RowCount * W.ColumnCount; i ++) {
            W.Elements[i].Data -= W.Elements[i].Grad * lr;
            W.Elements[i].Grad = 0.0;
        }
        for (int i = 0; i < CIFAR_CLASSES; i++) {
            bias[i].Data -= bias[i].Grad * lr;
            bias[i].Grad = 0.0;
        }

        if (step == 0 || step == train_steps - 1) {
            LogArena(&weights_arena);
            LogArena(&scratch_arena);
            LogArena(&forward_arena);
        }
    
        memset((void*)forward_arena.Base, 0, forward_arena.Used);
        forward_arena.Used = 0;
        scratch_arena.Used = 0;

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
         
        Value* x = MemoryArenaPush(&forward_arena, sizeof(Value) * IMG_SIZE);
        for (int i = 0; i < IMG_SIZE; i++) {
            //x[i].Data = (f64)entry_view.ImageData[i] / 255.0; // normalize to [0..1]
            x[i].Data = ((f64)entry_view.ImageData[i] - 127.5) / 127.5; // ~ [-1, 1]
            x[i].Grad = 0;
            x[i].ChildCount = 0;
        }
        int y_idx = entry_view.Label;

        // forward
        // W @ x
        Value** out = MemoryArenaPush(&scratch_arena, sizeof(Value*) * CIFAR_CLASSES);
        Linear(&W, out, x, &forward_arena);
        for (int i = 0; i < CIFAR_CLASSES; i++) {
            Value* elem_with_bias = NewValue(&forward_arena);
            VAdd(elem_with_bias, out[i], &bias[i]);
            out[i] = elem_with_bias;
        }
        
        // calculate softmax and loss for correct label
        Value** probs = MemoryArenaPush(&scratch_arena, sizeof(Value*) * CIFAR_CLASSES);
        SoftMax(out, probs, CIFAR_CLASSES, &forward_arena, &scratch_arena);
        
        Value* y = probs[y_idx];
        Value* log_prob = NewValue(&forward_arena);
        VLog(log_prob, y);

        Value* loss = NewValue(&forward_arena);
        VNegative(loss, log_prob);

        loss_accum += loss->Data;

        int predicted_class = 0;
        for (int c = 1; c < CIFAR_CLASSES; c ++) {
            if (probs[c]->Data > probs[predicted_class]->Data) predicted_class = c;
        }
        if (predicted_class == y_idx) correct_predictions++;

        memset((void*)forward_arena.Base, 0, forward_arena.Used);
        forward_arena.Used = 0;
        scratch_arena.Used = 0;
    }

    printf("Test Accuracy: %.3f%% loss avg: %f\n", 
        100.0 * (f64)correct_predictions / (f64)ENTRIES_PER_BATCH,
        loss_accum / (f64)ENTRIES_PER_BATCH);

    
    return 0;
}