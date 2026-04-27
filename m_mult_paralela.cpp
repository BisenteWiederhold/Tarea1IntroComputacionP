#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;

// Genera una matriz aleatoria de n x n
Matrix random_matrix(int n) {
    Matrix M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand() % 10;
    return M;
}

// Implementación paralela de multiplicación por bloques (Cache-friendly)
void mult_bloques_parallel(const Matrix &A, const Matrix &B, Matrix &C, int n, int b) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b)
            for (int kk = 0; kk < n; kk += b)
                for (int i = ii; i < min(ii + b, n); i++)
                    for (int j = jj; j < min(jj + b, n); j++)
                        for (int k = kk; k < min(kk + b, n); k++)
                            C[i][j] += A[i][k] * B[k][j];
}

// Funciones auxiliares para Strassen
Matrix add(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2) if(n > 256)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix sub(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2) if(n > 256)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Algoritmo de Strassen Paralelo (Híbrido con bloques en caso base) [cite: 107]
Matrix strassen_parallel_recursive(const Matrix &A, const Matrix &B, int n) {
    // Umbral de corte para pasar a secuencial/bloques 
    if (n <= 256) {
        Matrix C(n, vector<double>(n));
        mult_bloques_parallel(A, B, C, n, 64);
        return C;
    }

    int k = n / 2;
    Matrix A11(k, vector<double>(k)), A12(k, vector<double>(k)), A21(k, vector<double>(k)), A22(k, vector<double>(k));
    Matrix B11(k, vector<double>(k)), B12(k, vector<double>(k)), B21(k, vector<double>(k)), B22(k, vector<double>(k));

    #pragma omp parallel for collapse(2) if(n > 512)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j]; A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j]; A22[i][j] = A[i + k][j + k];
            B11[i][j] = B[i][j]; B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j]; B22[i][j] = B[i + k][j + k];
        }
    }

    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp task shared(M1) if(n > 512)
    M1 = strassen_parallel_recursive(add(A11, A22, k), add(B11, B22, k), k);
    #pragma omp task shared(M2) if(n > 512)
    M2 = strassen_parallel_recursive(add(A21, A22, k), B11, k);
    #pragma omp task shared(M3) if(n > 512)
    M3 = strassen_parallel_recursive(A11, sub(B12, B22, k), k);
    #pragma omp task shared(M4) if(n > 512)
    M4 = strassen_parallel_recursive(A22, sub(B21, B11, k), k);
    #pragma omp task shared(M5) if(n > 512)
    M5 = strassen_parallel_recursive(add(A11, A12, k), B22, k);
    #pragma omp task shared(M6) if(n > 512)
    M6 = strassen_parallel_recursive(sub(A21, A11, k), add(B11, B12, k), k);
    #pragma omp task shared(M7) if(n > 512)
    M7 = strassen_parallel_recursive(sub(A12, A22, k), add(B21, B22, k), k);

    #pragma omp taskwait

    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2) if(n > 512)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + k]     = M3[i][j] + M5[i][j];
            C[i + k][j]     = M2[i][j] + M4[i][j];
            C[i + k][j + k] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
    return C;
}

// Wrapper para manejar matrices de cualquier tamaño (Padding a potencia de 2) 
Matrix strassen_parallel(const Matrix &A, const Matrix &B, int n) {
    int m = 1;
    while (m < n) m *= 2;

    if (m == n) {
        Matrix res;
        #pragma omp parallel
        #pragma omp single
        res = strassen_parallel_recursive(A, B, n);
        return res;
    }

    Matrix A_pad(m, vector<double>(m, 0.0)), B_pad(m, vector<double>(m, 0.0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_pad[i][j] = A[i][j]; B_pad[i][j] = B[i][j];
        }
    }

    Matrix C_pad;
    #pragma omp parallel
    #pragma omp single
    C_pad = strassen_parallel_recursive(A_pad, B_pad, m);

    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = C_pad[i][j];

    return C;
}

int main() {
    vector<int> sizes = {256, 512, 1024};
    vector<int> threads = {1, 2, 4, 8}; // Cambiado a 8 según sugerencia del enunciado 
    int b = 64;

    ofstream file("resultados_speedup.csv");
    file << "n,threads,bloques_time,strassen_time,speedup_bloques,eficiencia_bloques,speedup_strassen,eficiencia_strassen\n";

    for (int n : sizes) {
        cout << "\n--- Tamaño n = " << n << " ---" << endl;
        Matrix A = random_matrix(n), B = random_matrix(n);
        double T1_b, T1_s;

        // Caso base p=1 para calcular speedup
        omp_set_num_threads(1);
        Matrix C(n, vector<double>(n));
        auto s1 = high_resolution_clock::now();
        mult_bloques_parallel(A, B, C, n, b);
        T1_b = duration<double>(high_resolution_clock::now() - s1).count();

        auto s2 = high_resolution_clock::now();
        strassen_parallel(A, B, n);
        T1_s = duration<double>(high_resolution_clock::now() - s2).count();

        for (int t : threads) {
            omp_set_num_threads(t);
            auto start_b = high_resolution_clock::now();
            mult_bloques_parallel(A, B, C, n, b);
            double tb = duration<double>(high_resolution_clock::now() - start_b).count();

            auto start_s = high_resolution_clock::now();
            strassen_parallel(A, B, n);
            double ts = duration<double>(high_resolution_clock::now() - start_s).count();

            file << n << "," << t << "," << tb << "," << ts << "," << T1_b/tb << "," << (T1_b/tb)/t << "," << T1_s/ts << "," << (T1_s/ts)/t << "\n";
            cout << "T=" << t << " | Bloques: " << tb << "s (S=" << T1_b/tb << ") | Strassen: " << ts << "s (S=" << T1_s/ts << ")" << endl;
        }
    }
    file.close();
    return 0;
}