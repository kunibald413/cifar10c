#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

/* 

https://cave.cs.toronto.edu/kriz/cifar.html

formatted as follows:
<1 x label><3072 x pixel>
...
<1 x label><3072 x pixel>

In other words, the first byte is the label of the first image, which is a number in the range 0-9. 
The next 3072 bytes are the values of the pixels of the image. 
The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. 
The values are stored in row-major order, so the first 32 bytes 
are the red channel values of the first row of the image.

Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. 
Therefore each file should be exactly 30730000 bytes long.

There is another file, called batches.meta.txt. 
This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. 
It is merely a list of the 10 class names, one per row. 
The class name on row i corresponds to numeric label i.

*/

typedef uint8_t   u8;
typedef uint8_t byte;
typedef uint32_t u32;
typedef int32_t  i32;

#define IMG_W 32
#define IMG_H 32
#define IMG_PIXELS (IMG_W * IMG_H)          /* 1024 */
#define IMG_SIZE (3 * IMG_PIXELS)
#define ENTRY_SIZE (1 + IMG_SIZE)    /* 3073  */
#define ENTRIES_PER_BATCH 10000
#define BYTES_PER_BATCH (ENTRIES_PER_BATCH * ENTRY_SIZE)
#define TRAIN_BATCHES 5

typedef struct CifarBatch {
    int EntryCount;
    void* RawData;
} CifarBatch;

typedef struct CifarEntryView {
    /* the first byte is the label of the first image, which is a number in the range 0-9.  */
    u8 Label;
    /*  
        View pointer to image data.
        The first 1024 bytes are the red channel values, 
        the next 1024 the green, and the final 1024 the blue. 
        The values are stored in row-major order, so the first 32 bytes 
        are the red channel values of the first row of the image. 
    */
    byte* ImageData;
    i32 Index;
} CifarEntryView;

CifarBatch LoadCifarBatch(const char* path) {
    void* raw_data = calloc(ENTRIES_PER_BATCH, ENTRY_SIZE);
    if (raw_data == NULL) {
        printf("faild to allocate %d bytes for batch!\n", BYTES_PER_BATCH);
        exit(1);
    }
    CifarBatch batch;
    batch.EntryCount = ENTRIES_PER_BATCH;
    batch.RawData = raw_data;

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); exit(1); }

    if (fread(batch.RawData, BYTES_PER_BATCH, 1, fp) != 1) {
        fprintf(stderr, "failed to read batch %s\n", path);
        fclose(fp);
        exit(1);
    }
    fclose(fp);

    return batch;
}

void DestroyCifarBatch(CifarBatch batch) {
    free(batch.RawData);
    batch.EntryCount = 0;
    batch.RawData = NULL;
}

CifarEntryView GetEntryView(CifarBatch cifar, i32 entry_index) {
    if (entry_index < 0 || entry_index >= cifar.EntryCount) {
        printf("Invalid index: %d\n", entry_index);
        exit(1);
    }

    byte* entryptr = (byte*)cifar.RawData + entry_index * ENTRY_SIZE;
    CifarEntryView entry;
    entry.ImageData = entryptr + 1;
    entry.Label = (u8)(*entryptr);
    entry.Index = entry_index;
    return entry;
}

int SaveEntryToPPM(CifarEntryView view) {
    char outname[64];
    snprintf(outname, sizeof(outname), "img_%06d_l_%d.ppm", view.Index, view.Label);

    FILE *out = fopen(outname, "wb");
    if (!out) { perror("fopen out"); return 1; }

    fprintf(out, "P6\n%d %d\n255\n", IMG_W, IMG_H);

    const byte *R = view.ImageData + 0 * IMG_PIXELS;
    const byte *G = view.ImageData + 1 * IMG_PIXELS;
    const byte *B = view.ImageData + 2 * IMG_PIXELS;

    byte pixel[IMG_SIZE];
    for (int i = 0; i < IMG_PIXELS; i++) {
        pixel[3*i + 0] = R[i];
        pixel[3*i + 1] = G[i];
        pixel[3*i + 2] = B[i];
    }

    fwrite(pixel, 1, sizeof(pixel), out);
    fclose(out);

    printf("wrote %s\n", outname);

    return 0;
}

void Example() {
    srand((u32)1234);
    const char *data_path = "data/cifar-10-batches-bin/data_batch_1.bin";

    CifarBatch batch = LoadCifarBatch(data_path);

    CifarEntryView entry = GetEntryView(batch, 2);
    printf("entry 2 label: %d\n", entry.Label);

    for (int i = 0; i < ENTRIES_PER_BATCH; i++) {
        CifarEntryView e = GetEntryView(batch, i);
        // printf("entry %d label: %d\n", i, e.Label);
        assert(e.Label >= 0 && e.Label <= 9 && "Label must be between 0 and 9");
    }

    for (int i = 0; i < 10; i ++) {
        int index = rand() % ENTRIES_PER_BATCH;
        printf("chosen index: %d\n", index);
        CifarEntryView random_entry = GetEntryView(batch, index);
        printf("random entry %d label: %d\n", random_entry.Index, random_entry.Label);
    }


    DestroyCifarBatch(batch);
}

#ifdef CIFAR_10_RUN_EXAMPLE
int main(int argc, char *argv[]) {
    Example();
    return 0;
}
#endif // CIFAR_10_RUN_EXAMPLE