__global__ void funkcija(float *rez, float* m, int *V, int *k){
    int t= (blockDim.x*blockDim.y)*threadIdx.z+    (threadIdx.y*blockDim.x)+(threadIdx.x);
}