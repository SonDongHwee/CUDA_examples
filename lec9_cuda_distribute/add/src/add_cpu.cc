
void vecAdd_omp(float* A, float* B, float* C, long long int n) {
#pragma omp parallel for
  for(int i=0;i<n;i++) {
    C[i] = A[i] + B[i];
  }
}

