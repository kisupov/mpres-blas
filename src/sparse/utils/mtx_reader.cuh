/*
 *  Utilities for reading Matrix Market coordinate format (mtx) and building the CSR, JAD, ELL, and DIA structures
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


#ifndef MPRES_MTX_READER_CUH
#define MPRES_MTX_READER_CUH

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <types.cuh>
#include "csr_utils.cuh"
#include <common.cuh>
#include <cassert>

using namespace std;

//TODO ни один конвертер + функция считывания параметров матрицы не учитывает нули, которые могут быть внутри .mtx файла

/*!
 * Collects sparse matrix statistics from mtx file
 * Input:
 * @param filename - path to the file with the matrix
 * Output:
 * @param m - number of rows in matrix
 * @param n - number of columns in matrix
 * @param lines - total number of lines with data
 * @param maxnzr - maximum number of nonzeros per row in the matrix
 * @param nzmd - number of nonzeros in the main diagonal of the matrix
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param datatype - type of data according to the matrix market format - real, integer, binary
 */
void collect_mtx_stats(const char *filename, int &m, int &n, int &lines, int &maxnzr, int &nzmd, bool &symmetric, string &datatype) {

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
    file >> m >> n >> lines;

    // Array for storing the number of non-zero elements in each row
    // For zero-initializing the array, we use value initialization in the constructor initialization list
    int *nonZeros = new int[m]();
    nzmd = 0;

    // Iterating over the matrix
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        nonZeros[(row - 1)] = nonZeros[(row - 1)] + 1;
        if (row == col){
            nzmd++;
        }
        if (symmetric && (row != col)) {
            nonZeros[(col - 1)] = nonZeros[(col - 1)] + 1;
        }
    }
    maxnzr = *std::max_element(nonZeros, nonZeros + m);
    delete[] nonZeros;
    file.close();
}

/*!
 * Calculates the number of nonzero diagonals in the sparse matrix
 * @param filename - path to the file with the matrix
 * @param lines - total number of lines with data
 * @param ndiag - number of nonzero diagonals
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 */
int calc_ndiag(const char filename[], const int lines, bool symmetric) {
    std::ifstream file(filename);
    while (file.peek() == '%') file.ignore(2048, '\n');
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::vector<int> diagOffsets(lines);
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        diagOffsets[l] = col-row;
    }
    std::sort(diagOffsets.begin(), diagOffsets.end());
    diagOffsets.erase(std::unique(diagOffsets.begin(), diagOffsets.end()), diagOffsets.end());
    int ndiag = (int) diagOffsets.size();
    if (symmetric) {
        for (int i = ndiag-2; i > -1; i--) {
            diagOffsets.push_back(-diagOffsets[i]);
        }
        ndiag = (int) diagOffsets.size();
    }
    file.close();
    diagOffsets.clear();
    diagOffsets.shrink_to_fit();
    return ndiag;
}

//сортирует 3 массива одинакового размера относительно массива ia (COO rows) по возрастанию
static void sort_coo_rows(const int nnz, double* as, int* ia, int* ja) {
    struct coo {
        double data;
        int cols;
        int rows;
    };
    vector <coo> x(nnz);
    for (int i = 0; i < nnz; i++) {
        x[i].data = as[i];
        x[i].rows = ia[i];
        x[i].cols = ja[i];
    }
    sort(x.begin(), x.end(), [] (const coo a, const coo b) {
        return a.rows < b.rows;
    });
    for (int i = 0; i < nnz; i++) {
        as[i] = x[i].data;
        ia[i] = x[i].rows;
        ja[i] = x[i].cols;
    }
}

//сортирует 2 массива одинакового размера относительно массива ia по убыванию
static void sort_perm_rows(const int nnz, int* ia, int* ja) {
    struct coo {
        int cols;
        int rows;
    };
    vector <coo> x(nnz);
    for (int i = 0; i < nnz; i++) {
        x[i].rows = ia[i];
        x[i].cols = ja[i];
    }
    sort(x.begin(), x.end(), [] (const coo a, const coo b) {
        return a.rows > b.rows;
    });
    for (int i = 0; i < nnz; i++) {
        ia[i] = x[i].rows;
        ja[i] = x[i].cols;
    }
}

//метод, формирующий массив смещений строк IRP (row start pointers array) для CSR-формата из массива IA (COO rows)
static void make_irp_array(const int m, const int nnz, int *ia, int *irp) {
    int p = 0;
    for (int i = 0; i < (m + 1); i++) {
        while (i > ia[p] && p < nnz) {
            p++;
        }
        irp[i] = p;
    }
}

/*!
 * Converts a sparse matrix to the COO format
 * @param filename - path to the file with the matrix
 * @param m - number of rows in the matrix
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false

 * @param as - coefficients array (COO data): an array of size lines containing a matrix data in the COO format (output parameter)
 * @param ia - row indices array (COO rows): an array of size lines containing the row indices (output parameter)
 * @param ja - column indices array (COO cols): an array of size lines containing the column indices (output parameter)
 */
void convert_to_coo(const char filename[], const int m, const int lines, bool symmetric, double *as, int *ia, int *ja) {

    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int j = lines;
    int nzmd = 0;

    // Iterating over the matrix
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        as[l] = fileData;
        ja[l] = col - 1;
        ia[l] = row - 1;
        if (col == row) {
            nzmd++;
        }
        if (symmetric && (row != col)) {
            as[j] = fileData;
            ja[j] = row - 1;
            ia[j] = col - 1;
            j++;
        }
    }
    file.close();

    int nnz = 0;
    if (symmetric){
        nnz = (lines - nzmd) * 2 + nzmd;
    } else {
        nnz = lines;
    }

    sort_coo_rows(nnz, as, ia, ja);
}

/*!
 * Converts a sparse matrix to the CSR format
 * @param filename - path to the file with the matrix
 * @param m - number of rows in the matrix
 * @param nnz - number of nonzeros in the matrix
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param csr - reference to the CSR instance to be defined
 */
void build_csr(const char filename[], const int m, const int nnz, const int lines, const bool symmetric, csr_t &csr) {
    int *ia = new int[nnz]();
    convert_to_coo(filename, m, lines, symmetric, csr.as, ia, csr.ja);
    make_irp_array(m, nnz, ia, csr.irp);
    delete[] ia;
}

/*!
 * Converts a sparse matrix to the DIA format
 * @param filename - path to the file with the matrix
 * @param m - number of rows in the matrix
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param dia - reference to the DIA instance to be defined
 */
void build_dia(const char filename[], const int m, const int lines, const bool symmetric, dia_t & dia) {
    //читаем файл дважды в методе. Вместе с методом collect_mtx_stats получается 3
    std::ifstream file(filename);
    while (file.peek() == '%') file.ignore(2048, '\n');
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //Skip one line with the matrix properties
    std::vector<int> diagOffsets(lines);
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        diagOffsets[l] = col-row;
    }
    std::sort(diagOffsets.begin(), diagOffsets.end());
    diagOffsets.erase(std::unique(diagOffsets.begin(), diagOffsets.end()), diagOffsets.end());
    auto ndiag = (int) diagOffsets.size();
    if (symmetric) {
        for (int i = ndiag-2; i > -1; i--) {
            diagOffsets.push_back(-diagOffsets[i]);
        }
        ndiag = (int) diagOffsets.size();
    }
    dia.as = new double[m * ndiag]();
    dia.offset = new int[ndiag];
    for (int i = 0; i < ndiag; ++i) {
        dia.offset[i] = diagOffsets[i];
    }
    //again from beginning of file
    file.seekg(0, ios::beg);
    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (int l = 0; l < lines; ++l) {
        double fileData = 0.0;
        int row = 0, col = 0, index = 0;
        file >> row >> col >> fileData;
        index = (int) std::distance(diagOffsets.begin(), std::find(diagOffsets.begin(), diagOffsets.end(), (col-row)));
        dia.as[m * index + (row-1)] = fileData;
        if (symmetric) {
            index = (int) std::distance(diagOffsets.begin(), std::find(diagOffsets.begin(), diagOffsets.end(), (row-col)));
            dia.as[m * index + (col-1)] = fileData;
        }
    }
    file.close();
    diagOffsets.clear();
    diagOffsets.shrink_to_fit();
}


/*!
 * Converts a sparse matrix to the JAD format
 * @param filename - path to the file with the matrix
 * @param m - number of rows in the matrix
 * @param maxnzr - maximum number of nonzeros per row
 * @param nnz - number of nonzeros in the matrix
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param jad - reference to the JAD instance to be defined
 */
void build_jad(const char *filename, const int m, const int maxnzr, const int nnz, const int lines, const bool symmetric, jad_t &jad) {
    csr_t csr;
    csr_init(csr, m, nnz);
    auto *nonZeros = new int[m]();
    build_csr(filename, m, nnz, lines, symmetric, csr);
    for (int i = 0; i < m; ++i) {
        jad.perm[i] = i; // сеттим массив perm_rows значениями от 0 до m-1
        nonZeros[i] = csr.irp[i + 1] - csr.irp[i]; // получаем кол-во ненулевых элементов в i строке
    }
    //сортируем массив индексов строк относительно кол-ва ненулевых элементов в строке по убыванию
    sort_perm_rows(m, nonZeros, jad.perm);
    //вычисляем смещение по столбцам
    jad.jcp[0] = 0;
    int count = 0;
    for (int j = 0; j < maxnzr; ++j) {
        for (int i = 0; i < m; ++i) {
            if (nonZeros[i] > 0) {
                nonZeros[i]--;
                count++;
            } else {
                continue;
            }
        }
        jad.jcp[j+1] = count;
    }
    //сеттим меасивы as и ja в новом порядке
    int index = 0;
    int j = 0;
    while (index < nnz && j < maxnzr) {
        for (int i = 0; i < jad.jcp[j + 1] - jad.jcp[j]; i++) {
            jad.as[index] = csr.as[j + csr.irp[jad.perm[i]]];
            jad.ja[index] = csr.ja[j + csr.irp[jad.perm[i]]];
            index++;
        }
        j++;
    }
    csr_clear(csr);
    delete[] nonZeros;
}

/*!
 * Converts a sparse matrix to the ELLPACK format
 * @param filename - path to the file with the matrix
 * @param m - number of rows in the matrix
 * @param maxnzr - maximum number of nonzeros per row in the matrix
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param ell - reference to the ELLPACK instance to be defined
 */
void build_ell(const char filename[], const int m, const int maxnzr, const int lines, const bool symmetric, ell_t &ell) {

    //Set default values
    std::fill(ell.ja, ell.ja + m * maxnzr, -1);
    std::fill(ell.as, ell.as + m * maxnzr, 0);

    //Create stream
    std::ifstream file(filename);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');
    //Skip one line with the matrix properties
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int *colNum = new int[m]();

    // Iterating over the matrix
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        ell.as[colNum[(row - 1)] * m + (row - 1)] = fileData;
        ell.ja[colNum[(row - 1)] * m + (row - 1)] = (col - 1);
        colNum[row - 1]++;
        if (symmetric && (row != col)) {
            ell.as[colNum[(col - 1)] * m + (col - 1)] = fileData;
            ell.ja[colNum[(col - 1)] * m + (col - 1)] = (row - 1);
            colNum[col - 1]++;
        }
    }
    delete[] colNum;
    file.close();
}


/*!
 * Converts a sparse matrix to a regular dense matrix stored in column-major order
 * @param filename - path to the file with the matrix
 * @param m - number of rows
 * @param n -  number of columns
 * @param lines - total number of lines with data
 * @param symmetric - true if the input matrix is to be treated as symmetrical; otherwise false
 * @param matrix - reference to the dense matrix to be defined
 */
void build_dense(const char filename[], const int m, const int n, const int lines, const bool symmetric, double * matrix) {
    std::fill(matrix, matrix + m * n, 0);
    std::ifstream file(filename);
    while (file.peek() == '%') file.ignore(2048, '\n');
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (int l = 0; l < lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        matrix[(col - 1) * m + row - 1] = fileData;
        if (symmetric && (row != col)) {
            matrix[(row - 1) * m + col - 1] = fileData;
        }
    }
    file.close();
}


/*!
 * Reads double-precision vector from file
 * @param filename - path to the file with the array
 * @param n - size of vector
 * @param vector - reference to the memory to be written
 */
void read_vector(const char filename[], const int n, double * vector) {
    std::ifstream file(filename);
    for (int l = 0; l < n; l++) {
        file >> vector[l];
    }
    file.close();
}


#endif //MPRES_MTX_READER_CUH
