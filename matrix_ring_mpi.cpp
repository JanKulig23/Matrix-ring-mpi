// Kompilacja:  mpic++ matrix_ring_mpi.cpp -o matrix_ring_mpi
// Uruchomienie: mpirun --hostfile my_hosts -np 8 ./matrix_ring_mpi
// Tworzenie pliku z liczbą rdzeni : echo "localhost slots=8" > my_hosts

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using namespace std;

void printMatrix(const vector<double>& M, int N) {
    cout << fixed << setprecision(4);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << M[i * N + j] << "\t";
        }
        cout << "\n";
    }
    cout << defaultfloat;
}

vector<double> generateMatrix(int N) {
    vector<double> A(N * N);
    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 100) / 10.0 + 1.0; 
    }
    return A;
}

// Funkcja sekwencyjna (wykonywana tylko przez jeden proces dla porównania)
void invertSequential(vector<double> A, int N) {
    vector<double> I(N * N, 0.0);
    for (int i = 0; i < N; i++) I[i * N + i] = 1.0;

    for (int i = 0; i < N; i++) {
        double diag = A[i * N + i];
        for (int j = 0; j < N; j++) {
            A[i * N + j] /= diag;
            I[i * N + j] /= diag;
        }
        for (int k = 0; k < N; k++) {
            if (k == i) continue;
            double factor = A[k * N + i];
            for (int j = 0; j < N; j++) {
                A[k * N + j] -= factor * A[i * N + j];
                I[k * N + j] -= factor * I[i * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. TWORZENIE PIERŚCIENIA
    MPI_Comm cart_comm;
    int dims[1] = {world_size};
    int periods[1] = {1};
    int reorder = 1;      
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);

    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    // 2. USTALENIE SĄSIADÓW
    // left - sąsiad od którego odbieramy
    // right - sąsiad do którego wysyłamy
    int left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);

    // Weryfikacja topologii 
    if (rank == 0) {
        printf("Weryfikacja Ring (Shift)\n");
    }
    MPI_Barrier(cart_comm);
    printf("Rank %d: Odbieram od %d, Wysylam do %d\n", rank, left, right);
    MPI_Barrier(cart_comm);

    vector<int> test_sizes;
    if (rank == 0) {
        test_sizes = {3, 8, 104, 248, 504, 1000, 2000}; 
    }
    int num_tests = 7; 

    for (int t = 0; t < num_tests; t++) {
        int N = 0;
        if (rank == 0) N = test_sizes[t];

        MPI_Bcast(&N, 1, MPI_INT, 0, cart_comm);

        if (N % size != 0) {
            if (rank == 0) cerr << "Pominiecie N=" << N << endl;
            continue;
        }

        int rows_per_proc = N / size;
        int elements_per_proc = rows_per_proc * N;

        vector<double> global_A;
        vector<double> local_A(elements_per_proc);
        vector<double> local_I(elements_per_proc, 0.0);

        // Inicjalizacja lokalnego I
        int start_row = rank * rows_per_proc;
        for (int i = 0; i < rows_per_proc; i++) {
            local_I[i * N + (start_row + i)] = 1.0; 
        }

        double seq_duration = 0.0;

        //Część 1: PRZYGOTOWANIE I OBLICZENIA SEKWENCYJNE (TYLKO RANK 0)
        if (rank == 0) {
            cout << "\n=== N = " << N << " ===" << endl;
            
            if (N == 3) {
                global_A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
                cout << "Macierz wejsciowa (3x3):" << endl;
                printMatrix(global_A, N);
            } else {
                global_A = generateMatrix(N);
            }

            vector<double> A_seq_copy = global_A; 
            double t_seq_start = MPI_Wtime();
            invertSequential(A_seq_copy, N);
            double t_seq_end = MPI_Wtime();
            
            seq_duration = t_seq_end - t_seq_start;
            cout << "Sekwencyjnie (1 proces): " << seq_duration << " s" << endl;
        }

        //Część 2: OBLICZENIA RÓWNOLEGŁE (MPI RING)
        MPI_Barrier(cart_comm);
        double t_par_start = MPI_Wtime();

        // Rozesłanie danych (Scatter)
        MPI_Scatter(global_A.data(), elements_per_proc, MPI_DOUBLE,
                    local_A.data(), elements_per_proc, MPI_DOUBLE,
                    0, cart_comm);

        vector<double> pivot_row_A(N);
        vector<double> pivot_row_I(N);

        // Pętla algorytmu
        for (int k = 0; k < N; k++) {
            int owner_rank = k / rows_per_proc;
            int local_k = k % rows_per_proc;

            // Krok 1: Właściciel przygotowuje wiersz
            if (rank == owner_rank) {
                for (int j = 0; j < N; j++) {
                    pivot_row_A[j] = local_A[local_k * N + j];
                    pivot_row_I[j] = local_I[local_k * N + j];
                }
                double diag_val = pivot_row_A[k];
                for (int j = 0; j < N; j++) {
                    pivot_row_A[j] /= diag_val;
                    pivot_row_I[j] /= diag_val;
                }
                // Aktualizujemy u siebie
                for (int j = 0; j < N; j++) {
                    local_A[local_k * N + j] = pivot_row_A[j];
                    local_I[local_k * N + j] = pivot_row_I[j];
                }

                // Wyślij do prawego sąsiada
                MPI_Send(pivot_row_A.data(), N, MPI_DOUBLE, right, 0, cart_comm);
                MPI_Send(pivot_row_I.data(), N, MPI_DOUBLE, right, 1, cart_comm);
            } 
            else {
                // Krok 2: Pozostali odbierają od lewego sąsiada
                MPI_Recv(pivot_row_A.data(), N, MPI_DOUBLE, left, 0, cart_comm, MPI_STATUS_IGNORE);
                MPI_Recv(pivot_row_I.data(), N, MPI_DOUBLE, left, 1, cart_comm, MPI_STATUS_IGNORE);

                // Krok 3: Podaj dalej, jeśli sąsiad po prawej to nie właściciel
                if (right != owner_rank) {
                    MPI_Send(pivot_row_A.data(), N, MPI_DOUBLE, right, 0, cart_comm);
                    MPI_Send(pivot_row_I.data(), N, MPI_DOUBLE, right, 1, cart_comm);
                }
            }

            // Krok 4: Eliminacja Gaussa
            for (int i = 0; i < rows_per_proc; i++) {
                int global_current_row = start_row + i;
                if (global_current_row != k) {
                    double factor = local_A[i * N + k];
                    for (int j = 0; j < N; j++) {
                        local_A[i * N + j] -= factor * pivot_row_A[j];
                        local_I[i * N + j] -= factor * pivot_row_I[j];
                    }
                }
            }
        }

        // Zbieranie wyników
        vector<double> global_inv;
        if (rank == 0) global_inv.resize(N * N);

        MPI_Gather(local_I.data(), elements_per_proc, MPI_DOUBLE,
                   global_inv.data(), elements_per_proc, MPI_DOUBLE,
                   0, cart_comm);

        double t_par_end = MPI_Wtime();
        double par_duration = t_par_end - t_par_start;

        if (rank == 0) {
            if (N == 3) {
                cout << "Wynik (Macierz odwrotna - MPI):" << endl;
                printMatrix(global_inv, N);
                cout << "-----------" << endl;
            }
            // Wyświetlenie obu czasów
            cout << "MPI RING (" << size << " procesow): " << par_duration << " s" << endl;
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}