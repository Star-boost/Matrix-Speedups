#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in ROW MAJOR ORDER.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    // BE SURE TO COMMENT CODE LATER!!!!
    return mat->data[((mat->cols) * row + col)];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in ROW MAJOR ORDER.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    mat->data[((mat->cols) * row + col)] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if ((rows < 1) || (cols < 1)) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix *temp = malloc(sizeof(matrix));
    if (temp == NULL) {
        return -2;
    } 
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // calloc bug 
    temp->data = calloc(1, rows * cols * sizeof(double));
    if (temp->data == NULL) {
        return -2;
    }
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    temp->rows = rows;
    temp->cols = cols;
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    temp->parent = NULL;
    // 6. Set the `ref_cnt` field to 1.
    temp->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = temp;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if (mat == NULL) {
        return;
    }
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        } 
    } else {
        // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
        deallocate_matrix(mat->parent);
        free(mat);
    }
}


/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if ((rows < 1) || (cols < 1)) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix *temp = malloc(sizeof(matrix));
    if (temp == NULL) {
        return -2;
    }
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    temp->data = from->data + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    temp->rows = rows;
    temp->cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    temp->parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    from->ref_cnt += 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = temp;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in ROW MAJOR ORDER.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int i;
    // set 1
    __m256d val_arr = _mm256_set1_pd (val);
    for (i = 0; i < (mat->cols * mat->rows) / 4 * 4; i += 4) {
        // store u
        _mm256_storeu_pd(mat->data + i, val_arr);
    }
    int j;
    for (j = (mat->cols * mat->rows) / 4 * 4; j < (mat->cols * mat->rows); j++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int i;
    __m256d val;
    __m256d max;
    __m256d multiplied;
    // set a zero vector
    __m256d negative_ones = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (i = 0; i < (mat->cols * mat->rows) / 4 * 4; i += 4) {
        // load
        val = _mm256_loadu_pd(mat->data + i);
        // check if greater
        // return new vector of 4
        multiplied = _mm256_mul_pd (val, negative_ones);
        max = _mm256_max_pd (val, multiplied);
        // store that new vector of 4
        _mm256_storeu_pd(result->data + i, max);
    }
    // tail case
    int j;
    for (j = (mat->cols * mat->rows) / 4 * 4; j < (mat->cols * mat->rows); j++) {
        if (mat->data[j] >= 0) {
            result->data[j] = mat->data[j];
        } else {
            result->data[j] = -(mat->data[j]);
        }   
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    
    // Make 3 temp vectors
    //load into the first 2
    //add into the second
    //store that vector into corresponding result
    int i;
    __m256d sum;
    __m256d temp1;
    __m256d temp2;
    #pragma omp parallel for
    for (i = 0; i < (mat1->rows * mat1->cols) / 4 * 4; i += 4) {
        temp1 = _mm256_loadu_pd(mat1->data + i);
        temp2 = _mm256_loadu_pd(mat2->data + i);
        sum = _mm256_add_pd(temp1, temp2);
        _mm256_storeu_pd(result->data + i, sum);
    }
    int j;
    // Tail case
    for (j = (mat1->rows * mat1->cols) / 4 * 4; j < (mat1->rows * mat2->cols); j++) {
        result->data[j] = mat1->data[j] + mat2->data[j];
    }
    return 0;    

    // Naive approach
    // #pragma omp parallel for
    // for (int i = 0; i < (mat1->rows * mat2->cols); i++) {
    //     result->data[i] = mat1->data[i] + mat2->data[i];
    // }
    // return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}


/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    // Transposing mat2
    matrix *mat2_transpose;
    allocate_matrix(&mat2_transpose, mat2->cols, mat2->rows);
    int col, row;
    #pragma omp parallel for private(row)
    for (col = 0; col < mat2->cols; col++) {
        for (row = 0; row < mat2->rows; row++) {
            mat2_transpose->data[((mat2_transpose->cols) * col) + row] = mat2->data[((mat2->cols) * row) + col];
        }
    }
    __m256d sum = _mm256_set1_pd(0.0);
    int i, j, k;
    double full_sum = 0.0;
    double store_last_four[4];
    double *mat1_data = mat1->data;
    double *mat2_transpose_data = mat2_transpose->data;
    #pragma omp parallel for private(full_sum, j, k, store_last_four)
    for (i = 0; i < mat1->rows; i++) {
        for (j = 0; j < mat2_transpose->rows; j++) {
            for (k = 0; k < mat1->cols / 20 * 20; k += 20) {
                __m256d temp1 = _mm256_loadu_pd(mat1_data + (i * mat1->cols) + k);
                __m256d temp2 = _mm256_loadu_pd(mat2_transpose_data + (j * mat2_transpose->cols) + k);
                sum = _mm256_fmadd_pd(temp1, temp2, sum);
                __m256d temp3 = _mm256_loadu_pd(mat1_data + (i * mat1->cols) + k + 4);
                __m256d temp4 = _mm256_loadu_pd(mat2_transpose_data + (j * mat2_transpose->cols) + k + 4);
                sum = _mm256_fmadd_pd(temp3, temp4, sum);
                __m256d temp5 = _mm256_loadu_pd(mat1_data + (i * mat1->cols) + k + 8);
                __m256d temp6 = _mm256_loadu_pd(mat2_transpose_data + (j * mat2_transpose->cols) + k + 8);
                sum = _mm256_fmadd_pd(temp5, temp6, sum);
                __m256d temp7 = _mm256_loadu_pd(mat1_data + (i * mat1->cols) + k + 12);
                __m256d temp8 = _mm256_loadu_pd(mat2_transpose_data + (j * mat2_transpose->cols) + k + 12);
                sum = _mm256_fmadd_pd(temp7, temp8, sum);
                __m256d temp9 = _mm256_loadu_pd(mat1_data + (i * mat1->cols) + k + 16);
                __m256d temp10 = _mm256_loadu_pd(mat2_transpose_data + (j * mat2_transpose->cols) + k + 16);
                sum = _mm256_fmadd_pd(temp9, temp10, sum);
            }
            // Tail case
            // _mm256_storeu_pd(result->data + (result->cols * i + j), sum);
            _mm256_storeu_pd(store_last_four, sum);
            sum = _mm256_set1_pd(0.0);
            full_sum = store_last_four[0] + store_last_four[1] + store_last_four[2] + store_last_four[3];
            for (int m = (mat1->cols / 20 * 20); m < mat1->cols; m++) {
                full_sum += mat1->data[((mat1->cols * i) + m)] * mat2_transpose->data[((mat2_transpose->cols) * j) + m];
            }
            result->data[((result->cols) * i + j)] = full_sum;
            full_sum = 0.0;
        }
    }    
    deallocate_matrix(mat2_transpose);
    return 0;
}


/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    matrix *x;
    matrix *xtemp;
    matrix *y;
    matrix *ytemp;
    matrix *identity;
    allocate_matrix(&x, mat->rows, mat->cols);      
    allocate_matrix(&xtemp, mat->rows, mat->cols);     
    allocate_matrix(&y, mat->rows, mat->cols);     
    allocate_matrix(&ytemp, mat->rows, mat->cols);     
    allocate_matrix(&identity, mat->rows, mat->cols);   
    int i, j;
    #pragma omp parallel for private(j)  
    for (i = 0; i < mat->rows; i++) {
        for (j = 0; j < mat->cols; j++) {
            if (i == j) {
                identity->data[((mat->cols) * i + j)] = 1;
            } else {
                identity->data[((mat->cols) * i + j)] = 0;
            }
        }
    }  
    if (pow == 0) {
        memcpy(result->data, identity->data, mat->rows * mat->cols * sizeof(double));
        deallocate_matrix(x);
        deallocate_matrix(xtemp);
        deallocate_matrix(y);
        deallocate_matrix(ytemp);
        deallocate_matrix(identity);
        return 0;
    }
    if (pow == 1) {
        memcpy(result->data, mat->data, mat->rows * mat->cols * sizeof(double));   
        deallocate_matrix(x);
        deallocate_matrix(xtemp);
        deallocate_matrix(y);
        deallocate_matrix(ytemp);
        deallocate_matrix(identity);
        return 0;     
    }
    memcpy(x->data, mat->data, mat->rows * mat->cols * sizeof(double));
    memcpy(xtemp->data, mat->data, mat->rows * mat->cols * sizeof(double));
    memcpy(y->data, identity->data, identity->rows * identity->cols * sizeof(double));
    memcpy(ytemp->data, identity->data, identity->rows * identity->cols * sizeof(double));
    while (pow > 1) {
        if (pow % 2 == 0) {
            mul_matrix(xtemp, x, x);
            memcpy(x->data, xtemp->data, x->rows * x->cols * sizeof(double));
            // mul_matrix(x, xtemp, identity);
            pow = pow / 2;
        } else {
            mul_matrix(ytemp, x, y);
            memcpy(y->data, ytemp->data, y->rows * y->cols * sizeof(double));
            // mul_matrix(y, ytemp, identity);
            mul_matrix(xtemp, x, x);
            memcpy(x->data, xtemp->data, x->rows * x->cols * sizeof(double));
            // mul_matrix(x, xtemp, identity);
            pow = (pow - 1) / 2;
        }
    }
    mul_matrix(result, x, y);
    deallocate_matrix(x);
    deallocate_matrix(xtemp);
    deallocate_matrix(y);
    deallocate_matrix(ytemp);
    deallocate_matrix(identity);
    return 0;    
}
