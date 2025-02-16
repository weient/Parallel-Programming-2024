#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <immintrin.h>
#include <omp.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int* image;
int* draw_image;
//int next_row;
//int chunk_size;
//int start;
//int end;
//omp_lock_t lock;
//pthread_mutex_t mutex;

void cal_row_j_vec_16(int j) 
{
    double y0 = j * ((upper - lower) / height) + lower;
    double x_unit = (right - left) / width;
    __m512d y0_vec = _mm512_set1_pd(y0);
    __m512d four_vec = _mm512_set1_pd(4);

    for (int i = 0; i < width; i += 16)
    {
        int valid_pixels = (width - i) < 16 ? (width - i) : 16;
        int repeats_mem[16] = {0};
        __m512i repeats = _mm512_setzero_si512();
        __m512d x0_vec_f = _mm512_set_pd(
            (i+7) * x_unit + left,
            (i+6) * x_unit + left,
            (i+5) * x_unit + left,
            (i+4) * x_unit + left,
            (i+3) * x_unit + left,
            (i+2) * x_unit + left,
            (i+1) * x_unit + left,
            i * x_unit + left
        );
        __m512d x0_vec_b = _mm512_set_pd(
            (i+15) * x_unit + left, 
            (i+14) * x_unit + left,
            (i+13) * x_unit + left, 
            (i+12) * x_unit + left,
            (i+11) * x_unit + left, 
            (i+10) * x_unit + left,
            (i+9) * x_unit + left, 
            (i+8) * x_unit + left
        );
        __m512d x_vec_f = _mm512_setzero_pd();
        __m512d x_vec_b = _mm512_setzero_pd();
        __m512d y_vec_f = _mm512_setzero_pd();
        __m512d y_vec_b = _mm512_setzero_pd();
        __m512d xx_vec_f = _mm512_setzero_pd();
        __m512d xx_vec_b = _mm512_setzero_pd();
        __m512d yy_vec_f = _mm512_setzero_pd();
        __m512d yy_vec_b = _mm512_setzero_pd();
        __m512d xy_vec_f = _mm512_setzero_pd();
        __m512d xy_vec_b = _mm512_setzero_pd();
        __m512d ls_vec_f = _mm512_setzero_pd();
        __m512d ls_vec_b = _mm512_setzero_pd();
        __mmask16 diverge_mask = 0xFFFF;

        int count = 0;
        while(diverge_mask && count < iters)
        {
            repeats = _mm512_mask_add_epi32(repeats, diverge_mask, repeats, _mm512_set1_epi32(1));
            x_vec_f = _mm512_add_pd(_mm512_sub_pd(xx_vec_f, yy_vec_f), x0_vec_f);
            x_vec_b = _mm512_add_pd(_mm512_sub_pd(xx_vec_b, yy_vec_b), x0_vec_b);
            y_vec_f = _mm512_add_pd(_mm512_add_pd(xy_vec_f, xy_vec_f), y0_vec);
            y_vec_b = _mm512_add_pd(_mm512_add_pd(xy_vec_b, xy_vec_b), y0_vec);
            xx_vec_f = _mm512_mul_pd(x_vec_f, x_vec_f);
            xx_vec_b = _mm512_mul_pd(x_vec_b, x_vec_b);
            yy_vec_f = _mm512_mul_pd(y_vec_f, y_vec_f);
            yy_vec_b = _mm512_mul_pd(y_vec_b, y_vec_b);
            xy_vec_f = _mm512_mul_pd(x_vec_f, y_vec_f);
            xy_vec_b = _mm512_mul_pd(x_vec_b, y_vec_b);
            ls_vec_f = _mm512_add_pd(xx_vec_f, yy_vec_f);
            ls_vec_b = _mm512_add_pd(xx_vec_b, yy_vec_b);
            diverge_mask = _mm512_cmp_pd_mask(ls_vec_f, four_vec, _CMP_LT_OQ) | (_mm512_cmp_pd_mask(ls_vec_b, four_vec, _CMP_LT_OQ) << 8);
            count++;
        }
        _mm512_storeu_si512(repeats_mem, repeats);
        for (int r=0; r<valid_pixels; r++) draw_image[j * width + i + r] = (int)repeats_mem[r];
    }
}


/*void job()
{
    int j;
    for (;;)
    {
        omp_set_lock(&lock);
        j = next_row++;
        omp_unset_lock(&lock);
        if (j >= end) break;
        cal_row_j_vec_16(j); 
    }
}
*/
int main(int argc, char** argv) {

    assert(argc == 9);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    
    image = (int*)malloc(width * height * sizeof(int));
    draw_image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    assert(draw_image);

    #pragma omp parallel for schedule(dynamic)
        for (int j=rank; j<height; j+=size) cal_row_j_vec_16(j);


    MPI_Reduce(draw_image, image, height * width, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_png(filename, iters, width, height, image);
    }

    free(image);
    free(draw_image);
    MPI_Finalize();
}
