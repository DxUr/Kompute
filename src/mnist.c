#if 0
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>


typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    float** weights;
    float* biases;
} Conv2D;

typedef struct {
    int pooling_size;
    int stride;
    int padding;
} SubSampl;

typedef struct {
    int input_channels;
    int output_channels;
    float* weights;
    float* biases;
} FullyConnected;


typedef struct {
    float input[28 * 28];
    Conv2D conv1;
    SubSampl sub1;
    Conv2D conv2;
    SubSampl sub2;
    FullyConnected fc1;
    FullyConnected fc2;
    float output[10];
} LeNet;

void init_conv(Conv2D* conv) {
    conv->weights = calloc(conv->output_channels, sizeof(float*));
    for (int i = 0; i < conv->output_channels; i++) {
        conv->weights[i] = calloc(conv->kernel_size * conv->kernel_size, sizeof(float));
    }
    conv->biases = calloc(conv->output_channels, sizeof(float));
}

void rand_conv(Conv2D* conv) {
    for (int i = 0; i < conv->output_channels; i++) {
        for (int j = 0; j < conv->kernel_size * conv->kernel_size; j++) {
            conv->weights[i][j] = (float)rand() / RAND_MAX;
        }
        conv->biases[i] = (float)rand() / RAND_MAX;
    }
}

void init_fc(FullyConnected* fc) {
    fc->weights = calloc(fc->output_channels, sizeof(float));
    fc->biases = calloc(fc->output_channels, sizeof(float));
}

void rand_fc(FullyConnected* fc) {
    for (int i = 0; i < fc->output_channels; i++) {
        fc->weights[i] = (float)rand() / RAND_MAX;
        fc->biases[i] = (float)rand() / RAND_MAX;
    }
}

void free_conv(Conv2D* conv) {
    for (int i = 0; i < conv->output_channels; i++) {
        free(conv->weights[i]);
    }
    free(conv->weights);
    free(conv->biases);
}

void free_fc(FullyConnected* fc) {
    free(fc->weights);
    free(fc->biases);
}

void init_net(LeNet* net) {
    net->conv1.input_channels = 1;
    net->conv1.output_channels = 6;
    net->conv1.kernel_size = 5;
    net->conv1.stride = 1;
    net->conv1.padding = 0;
    init_conv(&net->conv1);

    net->sub1.pooling_size = 2;
    net->sub1.stride = 2;
    net->sub1.padding = 0;

    net->conv2.input_channels = 6;
    net->conv2.output_channels = 16;
    net->conv2.kernel_size = 5;
    net->conv2.stride = 1;
    net->conv2.padding = 0;
    init_conv(&net->conv2);

    net->sub2.pooling_size = 2;
    net->sub2.stride = 2;
    net->sub2.padding = 0;

    net->fc1.input_channels = 120;
    net->fc1.output_channels = 84;
    init_fc(&net->fc1);

    net->fc2.input_channels = 84;
    net->fc2.output_channels = 10;
    init_fc(&net->fc2);
}

void rand_net(LeNet* net) {
    rand_conv(&net->conv1);
    rand_conv(&net->conv2);
    rand_fc(&net->fc1);
    rand_fc(&net->fc2);
}

void infer(LeNet* net) {
    printf("Infering...\n TODO\n");
}

// this will train the network for a single input
void train_net(LeNet* net, float learning_rate, uint8_t* dataset, uint8_t* labels) {
    printf("Training network...\n TODO\n");
}

void test_net(LeNet* net, uint8_t* dataset, uint8_t* labels) {
    printf("Testing network...\n TODO\n");
}

void free_net(LeNet* net) {
    free_conv(&net->conv1);
    free_conv(&net->conv2);
    free_fc(&net->fc1);
    free_fc(&net->fc2);
}

void save_net(LeNet* net, char* path) {
    // TODO
}

void load_net(LeNet* net, char* path) {
    // TODO
}

int main() {
    // srand(time(NULL));

    FILE* fp = fopen("train-images-idx3-ubyte", "r");
    if (fp == NULL) {
        printf("Failed to open train-images-idx3-ubyte\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* mnist_data = malloc(size);
    fread(mnist_data, 1, size, fp);
    fclose(fp);

    fp = fopen("train-labels-idx1-ubyte", "r");
    if (fp == NULL) {
        printf("Failed to open train-labels-idx1-ubyte\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* mnist_labels = malloc(size);
    fread(mnist_labels, 1, size, fp);
    fclose(fp);
    // Done with mnist

    char path[256];
    sprintf(path, "model%ld.bin", time(NULL));

    LeNet net;    
    init_net(&net);
    rand_net(&net);
    train_net(&net, 0.01, mnist_data, mnist_labels);
    test_net(&net, mnist_data, mnist_labels);
    save_net(&net, path);
    free_net(&net);

    free(mnist_data);
    free(mnist_labels);

    printf("Done!\n");

    return 0;
}
#endif