__global__ void funkcija(float* m, int *V1, int *k1){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int V = V1[0];
    int k = k1[0];
    if (i < V && j < V){
        float t1 = m[i*V + k] + m[k*V + j];
        float t2 = m[i*V + j];

        m[i*V + j] = (t1 < t2) ? t1 : t2;
    }
}