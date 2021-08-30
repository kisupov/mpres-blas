/*
 *  Utilities for printing the structure of a sparse matrix
 *
 *  Copyright 2021 by Konstantin Isupov.
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
#ifndef MPRES_MTX_PRINT_H
#define MPRES_MTX_PRINT_H

/*!
 * Prints a sparse matrix represented in the ELLPACK format
 */
void print_ellpack(const int m, const int maxnzr, double *as, int *ja) {
    std::cout << std::endl << "AS:";
    for (int i = 0; i < m; i++) {
        std::cout << std::endl;
        for (int j = 0; j < maxnzr; j++) {
            std::cout << as[i + m * j] << "\t";
        }
    }
    std::cout << std::endl << "JA:";
    for (int i = 0; i < m; i++) {
        std::cout << std::endl;
        for (int j = 0; j < maxnzr; j++) {
            std::cout << ja[i + m * j] << "\t";
        }
    }
}

/*!
 * Prints a sparse matrix represented in the CSR format
 */
void print_csr(const int m, const int nnz, double *as, int *irp, int *ja) {
    std::cout << std::endl << "AS:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << as[i] << "\t";
    }

    std::cout << std::endl << "IRP:" << std::endl;
    for (int i = 0; i < m + 1; i++) {
        std::cout << irp[i] << "\t";
    }

    std::cout << std::endl << "JA:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << ja[i] << "\t";
    }
}

/*!
 * Prints a sparse matrix represented in the DIA format
 */
void print_dia(const int m, const int ndiag, double *as, int *offset) {
    std::cout << std::endl << "OFFSET:";
    std::cout << std::endl;
    for (int j = 0; j < ndiag; j++) {
        std::cout << offset[j] << "\t";
    }

    std::cout << std::endl << "AS:";
    for (int i = 0; i < m; i++) {
        std::cout << std::endl;
        for (int j = 0; j < ndiag; j++) {
            std::cout << as[i + m * j] << "\t";
        }
    }
    std::cout << std::endl;
}

/*!
 * Prints a sparse matrix represented in the JAD (JDS) format
 */
void print_jad(const int m, const int nnz, const int maxnzr, double *as, int *ja, int *jcp, int *perm_rows) {
    std::cout << std::endl << "JA:";
    std::cout << std::endl;
    for (int j = 0; j < nnz; j++) {
        std::cout << ja[j] << "\t";
    }

    std::cout << std::endl << "AS:";
    std::cout << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << as[i] << "\t";
    }

    std::cout << std::endl << "JCP:";
    std::cout << std::endl;
    for (int i = 0; i < maxnzr + 1; i++) {
        std::cout << jcp[i] << "\t";
    }

    std::cout << std::endl << "PERM_ROWS:";
    std::cout << std::endl;
    for (int i = 0; i < m; i++) {
        std::cout << perm_rows[i] << "\t";
    }
    std::cout << std::endl;
}

#endif //MPRES_MTX_PRINT_H
