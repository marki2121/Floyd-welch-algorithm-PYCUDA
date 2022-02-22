__global__ void funkcija(float *rez, float* m, int *k){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (m(i, k) + m(k, j) < m(i, j)){
        m(i, j) = m(i, k) + m(k, j);
    }

    rez = m;
}
