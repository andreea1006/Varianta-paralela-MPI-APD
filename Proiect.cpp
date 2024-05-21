#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    // Ințializarea MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Procesul master citește imaginea
    Mat image;
    if (rank == 0) {
        image = imread("poza3.png", IMREAD_COLOR);
        if (image.empty()) {
            cout << "Nu s-a putut încărca imaginea. Verifică dacă calea este corectă." << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Trimite dimensiunile imaginii de la procesul master la toate celelalte procese
    int rows, cols;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculează dimensiunea unei bucăți de imagine
    int chunk_size = rows / size;
    int start_row = rank * chunk_size;
    int end_row = (rank + 1) * chunk_size;

    // Trimite bucățile de imagine la fiecare proces
    Mat local_image(chunk_size, cols, CV_8UC3);
    MPI_Scatter(image.data, chunk_size * cols * 3, MPI_BYTE, local_image.data, chunk_size * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Măsurarea timpului de început (doar pentru procesare)
    auto start = high_resolution_clock::now();

    // Oglindește fiecare bucățică de imagine pe verticală
    for (int i = 0; i < local_image.rows; ++i) {
        for (int j = 0; j < local_image.cols / 2; ++j) {
            Vec3b& pixel1 = local_image.at<Vec3b>(i, j);
            Vec3b& pixel2 = local_image.at<Vec3b>(i, local_image.cols - 1 - j);
            swap(pixel1, pixel2);
        }
    }

    // Măsurarea timpului de final (doar pentru procesare)
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Adună bucățile de imagine oglindite
    Mat mirrored_image(rows, cols, CV_8UC3);
    MPI_Gather(local_image.data, chunk_size * cols * 3, MPI_BYTE, mirrored_image.data, chunk_size * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Procesul master salvează imaginea oglindită și afișează timpul de execuție
    if (rank == 0) {
        imwrite("output_poza3_mpi.png", mirrored_image);
        cout << "Imaginea a fost oglindită și salvată cu succes." << endl;
        cout << "Timpul de execuție pentru procesarea imaginii: " << duration.count() << " milisecunde." << endl;
    }

    // Finalizarea MPI
    MPI_Finalize();

    return 0;
}
