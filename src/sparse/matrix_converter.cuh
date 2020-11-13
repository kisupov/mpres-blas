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
#include "assert.h"
#include "mparray.cuh"

using namespace std;

/*!
 * Reads metadata from a file that contains a sparse matrix
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix (output parameter)
 * @param num_cols - number of columns in the matrix (output parameter)
 * @param num_lines - total number of lines with data (output parameter)
 * @param num_cols_per_row - maximum number of nonzeros per row in the matrix (output parameter)
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param datatype - type of data according to the matrix market format - real, integer, binary
 */
void read_matrix_properties(const char filename[], int &num_rows, int &num_cols, int &num_lines, int &num_cols_per_row, bool &symmetric, string &datatype) {

    //Create stream
    std::ifstream file(filename);

    //Read header
    string head, type, formats, dtype, symmetry;
    file >> head >> type >> formats >> dtype >> symmetry;

    //header checking
    assert(head == "%%MatrixMarket");
    assert((type == "matrix") || (type == "tensor"));
    assert((symmetry == "general") || (symmetry == "symmetric"));

    datatype = dtype;
    symmetric = (symmetry == "symmetric");

    file.seekg(0, ios::beg);

    // Ignore comments headers
    while (file.peek() == '%') {
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
        if (symmetric && (row != col)) {
            nonZeros[(col - 1)] = nonZeros[(col - 1)] + 1;
        }
    }
    num_cols_per_row = *std::max_element(nonZeros, nonZeros + num_rows);
    delete[] nonZeros;
    file.close();
}

/*!
 * Converts a sparse matrix to the ELLPACK format
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix
 * @param num_cols_per_row - maximum number of nonzeros per row in the matrix
 * @param num_lines - total number of lines with data
 * @param data - ELLPACK data: an array of size num_rows * num_cols_per_row containing a matrix in the ELLPACK format (output parameter)
 * @param indices - ELLPACK indices: an array of size num_rows * num_cols_per_row containing the indices of nonzero elements in the matrix (output parameter)
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 */
void convert_to_ellpack(const char filename[], const int num_rows, const int num_cols_per_row, const int num_lines, double *data, int *indices, bool symmetric) {

    //Set default values
    std::fill(indices, indices + num_rows * num_cols_per_row, -1);
    std::fill(data, data + num_rows * num_cols_per_row, 0);

    //Create stream
    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int *colNum = new int[num_rows]();

    // Iterating over the matrix
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        data[colNum[(row - 1)] * num_rows + (row - 1)] = fileData;
        indices[colNum[(row - 1)] * num_rows + (row - 1)] = (col - 1);
        colNum[row - 1]++;
        if (symmetric && (row != col)) {
            data[colNum[(col - 1)] * num_rows + (col - 1)] = fileData;
            indices[colNum[(col - 1)] * num_rows + (col - 1)] = (row - 1);
            colNum[col - 1]++;
        }
    }
    delete[] colNum;
    file.close();
}

/*!
 * Prints a sparse matrix represented in the ELLPACK format
 */
void print_ellpack(const int num_rows, const int num_cols_per_row, double *data, int *indices) {
    std::cout << std::endl << "data:";
    for (int i = 0; i < num_rows; i++) {
        std::cout << std::endl;
        for (int j = 0; j < num_cols_per_row; j++) {
            std::cout << data[i + num_rows * j] << "\t";
        }
    }
    std::cout << std::endl << "indices:";
    for (int i = 0; i < num_rows; i++) {
        std::cout << std::endl;
        for (int j = 0; j < num_cols_per_row; j++) {
            std::cout << indices[i + num_rows * j] << "\t";
        }
    }
}

//сортирует 3 массива одинакового размера относительно массива rows
void sortCOORows(double* data, int* cols, int* rows, const int nnz) {
    struct coo {
        double data;
        int cols;
        int rows;
    };
    vector <coo> x(nnz);
    for (int i = 0; i < nnz; i++) {
        x[i].data = data[i];
        x[i].cols = cols[i];
        x[i].rows = rows[i];
    }
    sort(x.begin(), x.end(), [] (const coo a, const coo b) {
        return a.rows < b.rows;
    });
    for (int i = 0; i < nnz; i++) {
        data[i] = x[i].data;
        cols[i] = x[i].cols;
        rows[i] = x[i].rows;
    }
}

/*!
 * Converts a sparse matrix to the COO format
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix
 * @param num_lines - total number of lines with data
 * @param data - COO data: an array of size num_lines containing a matrix data in the COO format (output parameter)
 * @param rows - COO rows: an array of size num_lines containing the row indices (output parameter)
 * @param cols - COO cols: an array of size num_lines containing the col indices (output parameter)
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 */
void convert_to_coo(const char filename[], const int num_rows, const int num_lines, double *data, int *rows, int *cols, bool symmetric) {

    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int j = num_lines;

    // Iterating over the matrix
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        data[l] = fileData;
        cols[l] = col - 1;
        rows[l] = row - 1;
        if (symmetric && (row != col)) {
            data[j] = fileData;
            cols[j] = row - 1;
            rows[j] = col - 1;
            j++;
        }
    }
    file.close();

    int nnz = 0;
    if (symmetric){
        nnz = (num_lines - num_rows) * 2 + num_rows;
    } else {
        nnz = num_lines;
    }

    sortCOORows(data, cols, rows, nnz);
}

//метод, формирующий массив csr-offsets из массива coo-rows
void makeOffset(int nnz, int num_rows, int *rows, int *offset) {
    int p = 0;
    for (int i = 0; i < (num_rows + 1); i++) {
        while (i > rows[p] && p < nnz) {
            p++;
        }
        offset[i] = p;
    }
}

/*!
 * Converts a sparse matrix to the CSR format
 * @param filename - path to the file with the matrix
 * @param num_rows - number of rows in the matrix
 * @param num_lines - total number of lines with data
 * @param data - CSR data: an array of size num_lines containing a matrix data in the CSR format (output parameter)
 * @param offsets - CSR offsets: an array of size num_rows + 1 containing the offset of i-th row in offsets[i] (output parameter)
 * @param cols - CSR cols: an array of size num_lines containing the col indices (output parameter)
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 */
void convert_to_csr(const char filename[], const int num_rows, const int num_lines, double *data, int *offsets, int *cols, bool symmetric) {
    int nnz = 0;
    if (symmetric){
        nnz = (num_lines - num_rows) * 2 + num_rows;
    } else {
        nnz = num_lines;
    }
    int * rows = new int[nnz]();

    convert_to_coo(filename, num_rows, num_lines, data, rows, cols, symmetric);
    makeOffset(nnz, num_rows, rows, offsets);

    delete[] rows;
}

/*!
 * Prints a sparse matrix represented in the CSR format
 */
void print_csr(const int num_rows, const int nnZ, double *data, int *offsets, int *cols) {
    std::cout << std::endl << "data:" << std::endl;
    for (int i = 0; i < nnZ; i++) {
        std::cout << data[i] << "\t";
    }

    std::cout << std::endl << "offsets:" << std::endl;
    for (int i = 0; i < num_rows + 1; i++) {
        std::cout << offsets[i] << "\t";
    }

    std::cout << std::endl << "cols:" << std::endl;
    for (int i = 0; i < nnZ; i++) {
        std::cout << cols[i] << "\t";
    }
}

#endif //MATRIX_CONVERTER_CUH
