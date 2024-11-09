#include <iostream>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

__global__ void forward_pass(float *input, float *weights1, float *bias1, float *weights2, float *bias2, float *output) {
    int idx = threadIdx.x;

    // Hidden layer
    float hidden[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden[i] = 0.0f;
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden[i] += input[j] * weights1[j * HIDDEN_SIZE + i];
        }
        hidden[i] += bias1[i];
        hidden[i] = max(0.0f, hidden[i]); // ReLU activation
    }

    // Output layer
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            output[i] += hidden[j] * weights2[j * OUTPUT_SIZE + i];
        }
        output[i] += bias2[i];
    }
}

int main() {
    // Allocate host memory
    float *h_input = new float[INPUT_SIZE];
    float *h_weights1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *h_bias1 = new float[HIDDEN_SIZE];
    float *h_weights2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float *h_bias2 = new float[OUTPUT_SIZE];
    float *h_output = new float[OUTPUT_SIZE];

    // Initialize input and weights (for simplicity, using random values)
    for (int i = 0; i < INPUT_SIZE; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) h_weights1[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < HIDDEN_SIZE; ++i) h_bias1[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i) h_weights2[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < OUTPUT_SIZE; ++i) h_bias2[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_input, *d_weights1, *d_bias1, *d_weights2, *d_bias2, *d_output;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    forward_pass<<<1, 1>>>(d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << "Output[" << i << "]: " << h_output[i] << std::endl;
    }

    // Free memory
    delete[] h_input;
    delete[] h_weights1;
    delete[] h_bias1;
    delete[] h_weights2;
    delete[] h_bias2;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_output);

    return 0;
}