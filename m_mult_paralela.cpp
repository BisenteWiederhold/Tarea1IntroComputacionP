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

// Matriz aleatoria
Matrix random_matrix(int n) {
    Matrix M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand() % 10;
    return M;
}

// Bloques secuencial
void mult_bloques(const Matrix &A, const Matrix &B, Matrix &C, int n, int b) {

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0;

    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b)
            for (int kk = 0; kk < n; kk += b)
                for (int i = ii; i < min(ii + b, n); i++)
                    for (int j = jj; j < min(jj + b, n); j++)
                        for (int k = kk; k < min(kk + b, n); k++)
                            C[i][j] += A[i][k] * B[k][j];
}

// Bloques paralelo
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

// Suma / resta
Matrix add(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix sub(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Strassen secuencial
Matrix strassen(const Matrix &A, const Matrix &B, int n) {

    if (n <= 64) {
        Matrix C(n, vector<double>(n));
        mult_bloques(A, B, C, n, 64);
        return C;
    }

    int k = n / 2;

    Matrix A11(k, vector<double>(k)), A12(k, vector<double>(k)),
           A21(k, vector<double>(k)), A22(k, vector<double>(k));

    Matrix B11(k, vector<double>(k)), B12(k, vector<double>(k)),
           B21(k, vector<double>(k)), B22(k, vector<double>(k));

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j];
            A22[i][j] = A[i + k][j + k];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j];
            B22[i][j] = B[i + k][j + k];
        }

    auto M1 = strassen(add(A11, A22, k), add(B11, B22, k), k);
    auto M2 = strassen(add(A21, A22, k), B11, k);
    auto M3 = strassen(A11, sub(B12, B22, k), k);
    auto M4 = strassen(A22, sub(B21, B11, k), k);
    auto M5 = strassen(add(A11, A12, k), B22, k);
    auto M6 = strassen(sub(A21, A11, k), add(B11, B12, k), k);
    auto M7 = strassen(sub(A12, A22, k), add(B21, B22, k), k);

    Matrix C(n, vector<double>(n));

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + k]     = M3[i][j] + M5[i][j];
            C[i + k][j]     = M2[i][j] + M4[i][j];
            C[i + k][j + k] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }

    return C;
}

// Strassen paralelo
Matrix strassen_parallel(const Matrix &A, const Matrix &B, int n) {

    if (n <= 64) {
        Matrix C(n, vector<double>(n));
        mult_bloques_parallel(A, B, C, n, 64);
        return C;
    }

    int k = n / 2;

    Matrix A11(k, vector<double>(k)), A12(k, vector<double>(k)),
           A21(k, vector<double>(k)), A22(k, vector<double>(k));

    Matrix B11(k, vector<double>(k)), B12(k, vector<double>(k)),
           B21(k, vector<double>(k)), B22(k, vector<double>(k));

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j];
            A22[i][j] = A[i + k][j + k];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j];
            B22[i][j] = B[i + k][j + k];
        }

    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp parallel sections
    {
        #pragma omp section
        M1 = strassen_parallel(add(A11, A22, k), add(B11, B22, k), k);

        #pragma omp section
        M2 = strassen_parallel(add(A21, A22, k), B11, k);

        #pragma omp section
        M3 = strassen_parallel(A11, sub(B12, B22, k), k);

        #pragma omp section
        M4 = strassen_parallel(A22, sub(B21, B11, k), k);

        #pragma omp section
        M5 = strassen_parallel(add(A11, A12, k), B22, k);

        #pragma omp section
        M6 = strassen_parallel(sub(A21, A11, k), add(B11, B12, k), k);

        #pragma omp section
        M7 = strassen_parallel(sub(A12, A22, k), add(B21, B22, k), k);
    }

    Matrix C(n, vector<double>(n));

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + k]     = M3[i][j] + M5[i][j];
            C[i + k][j]     = M2[i][j] + M4[i][j];
            C[i + k][j + k] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }

    return C;
}


// Main
int main() {

    vector<int> sizes = {256, 512, 1024};
    vector<int> threads = {1, 2, 4, 8};
    int b = 64;

    ofstream file("resultados_paralelo.csv");
    file << "n,threads,bloques_par, strassen_par\n";

    for (int n : sizes) {

        Matrix A = random_matrix(n);
        Matrix B = random_matrix(n);

        for (int t : threads) {

            omp_set_num_threads(t);

            Matrix C(n, vector<double>(n));

            // bloques paralelo
            auto start = high_resolution_clock::now();
            mult_bloques_parallel(A, B, C, n, b);
            auto end = high_resolution_clock::now();
            double t_bloques = duration<double>(end - start).count();

            // Strassen paralelo
            start = high_resolution_clock::now();
            Matrix C2 = strassen_parallel(A, B, n);
            end = high_resolution_clock::now();
            double t_strassen = duration<double>(end - start).count();

            cout << "n=" << n << " threads=" << t << endl;
            cout << "Bloques_par: " << t_bloques << " s\n";
            cout << "Strassen_par: " << t_strassen << " s\n\n";

            file << n << "," << t << ","
                 << t_bloques << ","
                 << t_strassen << "\n";
        }
    }

    file.close();
    cout << "Datos guardados en resultados_paralelo.csv\n";

    return 0;
}