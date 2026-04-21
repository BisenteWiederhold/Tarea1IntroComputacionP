#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;

// Crear matriz aleatoria

Matrix random_matrix(int n) {
    Matrix M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand() % 10;
    return M;
}

// Multiplicación clasica

void mult_clasica(const Matrix &A, const Matrix &B, Matrix &C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// Multiplicación por bloques
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

// Suma y resta
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

// Strassen
Matrix strassen(const Matrix &A, const Matrix &B, int n) {

    if (n <= 64) {
        Matrix C(n, vector<double>(n));
        mult_clasica(A, B, C, n);
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

// Medir tiempo
double medir(void (*func)(const Matrix&, const Matrix&, Matrix&, int),
             const Matrix &A, const Matrix &B, Matrix &C, int n) {

    auto start = high_resolution_clock::now();
    func(A, B, C, n);
    auto end = high_resolution_clock::now();

    return duration<double>(end - start).count();
}

int main() {

    vector<int> sizes = {128, 256, 512, 1024}; // puedes agregar 2048 si aguanta
    vector<int> block_sizes = {16, 32, 64, 128};

    int repeticiones = 5; 

    ofstream file("resultados.csv");
    file << "n,b,clasica,bloques,strassen\n";

    for (int n : sizes) {

        cout << "\n========================\n";
        cout << "Tamaño n = " << n << endl;

        Matrix A = random_matrix(n);
        Matrix B = random_matrix(n);

        for (int b : block_sizes) {

            double total_clasica = 0;
            double total_bloques = 0;
            double total_strassen = 0;

            for (int r = 0; r < repeticiones; r++) {

                Matrix C(n, vector<double>(n));

                // Clásica
                auto start = high_resolution_clock::now();
                mult_clasica(A, B, C, n);
                auto end = high_resolution_clock::now();
                total_clasica += duration<double>(end - start).count();

                //Bloques 
                start = high_resolution_clock::now();
                mult_bloques(A, B, C, n, b);
                end = high_resolution_clock::now();
                total_bloques += duration<double>(end - start).count();

                //Strassen 
                start = high_resolution_clock::now();
                Matrix C2 = strassen(A, B, n);
                end = high_resolution_clock::now();
                total_strassen += duration<double>(end - start).count();
            }

            double avg_clasica = total_clasica / repeticiones;
            double avg_bloques = total_bloques / repeticiones;
            double avg_strassen = total_strassen / repeticiones;

            // Mostrar
            cout << "b = " << b << endl;
            cout << "Clasica: " << avg_clasica << " s\n";
            cout << "Bloques: " << avg_bloques << " s\n";
            cout << "Strassen: " << avg_strassen << " s\n\n";

            // Guardar
            file << n << "," << b << ","
                 << avg_clasica << ","
                 << avg_bloques << ","
                 << avg_strassen << "\n";
        }
    }

    file.close();
    cout << "Resultados guardados en resultados.csv\n";

    return 0;
}