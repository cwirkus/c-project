#include <string.h>

void mat_add(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* i-k-j loop order: keeps inner loop stride-1 on both b and out (cache-friendly) */
void mat_mul(const float *a, const float *b, float *out, int m, int k, int n) {
    memset(out, 0, (size_t)(m * n) * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++)
                out[i*n + j] += a[i*k + p] * b[p*n + j];
}

void mat_relu(float *a, int n) {
    for (int i = 0; i < n; i++)
        if (a[i] < 0.0f) a[i] = 0.0f;
}
