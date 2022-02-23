__global__ void funkcija(float *rez, float* m, int *k, int* V){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < V && j < V){
        
    }

    rez = m;




}
