/*
 *  Sparse matrix helper routines
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
 *
 *  This file is part of the MPRES-BLAS library.
 *
 *  MPRES-BLAS is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  MPRES-BLAS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with MPRES-BLAS.  If not, see <https://www.gnu.org/licenses/>.
 */


#ifndef MATRIX_CONVERTER_CUH
#define MATRIX_CONVERTER_CUH

#include <fstream>
#include "mparray.cuh"
using namespace std;

/*!
 * Reads metadata from a file that contains a sparse matrix
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix (output parameter)
 * @param num_cols - number of columns in the matrix (output parameter)
 * @param num_lines - total number of lines with data (output parameter)
 * @param num_cols_per_row - maximum number of nonzeros in the matrix (output parameter)
 */
void read_matrix_properties(const char filename[], int &num_rows, int &num_cols, int &num_lines, int &num_cols_per_row) {

    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%'){
        file.ignore(2048, '\n');
    }

    // Read number of rows and columns
    file >> num_rows >> num_cols >> num_lines;

    // Array for storing the number of non-zero elements in each row
    // For zero-initializing the array, we use value initialization in the constructor initialization list
    int *nonZeros = new int[num_rows]();

    // Iterating over the matrix
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        nonZeros[(row - 1)] = nonZeros[(row - 1)] + 1;
    }
    num_cols_per_row = * std::max_element(nonZeros, nonZeros + num_rows);
    delete [] nonZeros;
    file.close();
}

//TODO: Предусмотреть предустановку массива индексов отрицательными числами, например -1,
// чтобы можно было исключить обращение к нулевому элементу вектора, если какие-то данные отсутствуют.

/*!
 * Converts a sparse matrix to the ELLPACK format
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix
 * @param num_cols - number of columns in the matrix
 * @param num_cols_per_row - maximum number of nonzeros in the matrix
 * @param data - ELLPACK data: an array of size num_rows * num_cols_per_row containing a matrix in the ELLPACK format
 * @param data - ELLPACK indices: an array of size num_rows * num_cols_per_row containing the indices of nonzero elements in the matrix
 */
void convert_to_ellpack(const char filename[], const int num_rows, const int num_lines, double *data, int *indices) {

    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int * colNum = new int[num_rows]();

    // Iterating over the matrix
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        data[colNum[(row - 1)] * num_rows + (row - 1)] = fileData;
        indices[colNum[(row - 1)] * num_rows + (row - 1)] = (col-1);
        colNum[row - 1]++;
    }
    delete [] colNum;
    file.close();
}

/*!
 * Prints a sparse matrix represented in the ELLPACK format
 */
void print_ellpack(const int num_rows, const int num_cols_per_row, double *data, int *indices){
    std::cout<<std::endl << "data:";
    for(int i = 0; i < num_rows; i++){
        std::cout<<std::endl;
        for(int j = 0; j < num_cols_per_row; j++){
            std::cout<<data[i + num_rows * j] << "\t";
        }
    }
    std::cout<<std::endl << "indices:";
    for(int i = 0; i < num_rows; i++){
        std::cout<<std::endl;
        for(int j = 0; j < num_cols_per_row; j++){
            std::cout<<indices[i + num_rows * j] << "\t";
        }
    }
}

#endif //MATRIX_CONVERTER_CUH
