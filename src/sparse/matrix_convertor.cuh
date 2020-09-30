/*
 *  Multiple-precision reduction CUDA kernels (sum, sum of absolute values, max value)
 *
 *  Copyright 2019, 2020 by Konstantin Isupov.
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


#ifndef MATRIX_CONVERTOR_CUH
#define MATRIX_CONVERTOR_CUH

#include <fstream>
#include "mparray.cuh"
using namespace std;

void create_ellpack_matrices(char filename[], mp_float_t *&data, int *&indices, int &num_rows, int &num_cols, int &num_cols_per_row) {

    std::ifstream file(filename);
    int num_lines = 0;

// Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

// Read number of rows and columns
    file >> num_rows >> num_cols >> num_lines;

// Create 2D array and fill with zeros
    int *nonZeros;
    nonZeros = new int[num_rows]();

// fill the matrix with data
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        nonZeros[(row - 1)] = nonZeros[(row - 1)] + 1;
    }

    num_cols_per_row = *std::max_element(nonZeros, nonZeros + num_rows);

    data = new mp_float_t[num_rows * (num_cols_per_row)];
    indices = new int[num_rows * (num_cols_per_row)]();

    for (int i = 0; i < num_rows * num_cols_per_row; ++i) {
        mp_set_d(&data[i], 0);
    }

    //курсор в начало
    file.seekg(0, ios::beg);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_rows >> num_cols >> num_lines;

    int * colNum = new int[num_rows]();

    //разобраться как заново считывать файл
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        mp_set_d(&data[colNum[(row - 1)] * num_rows + (row - 1)], fileData);
        indices[colNum[(row - 1)] * num_rows + (row - 1)] = (col-1);
        colNum[row - 1]++;
    }

    file.close();
}


#endif //MATRIX_CONVERTOR_CUH
