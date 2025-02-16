#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
//#include <nvtx3/nvToolsExt.h>

void choose_min(float* local, float* adj, int local_n, int adj_n, float* tmp)
{
    int i = 0, j = 0, cnt = 0;
    while (cnt < local_n)
    {
        if (j >= adj_n || (i < local_n && local[i] < adj[j])) tmp[cnt++] = local[i++];
        else tmp[cnt++] = adj[j++];
    }
}


void choose_max(float* local, float* adj, int local_n, int adj_n, float* tmp)
{
    int i = local_n - 1, j = adj_n - 1 , cnt = local_n - 1;
    while (cnt >= 0)
    {
        if (j < 0 || (i >= 0 && local[i] > adj[j])) tmp[cnt--] = local[i--];
        else tmp[cnt--] = adj[j--];
    }
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // convert total elements to int
    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    // calculate number of elements to deal with in this process
    int local_n = n / size;
    int s_n = local_n;
    int l_n = s_n + 1;
    int remainder = n % size;
    if (rank < remainder) local_n++; 
    
    // the offset of this process to read file from
    int offset = (s_n * rank + (rank < remainder ? rank : remainder)) * sizeof(float);
    MPI_File input_file, output_file;
    float* local_data = (float*) malloc(local_n * sizeof(float));
    float* local_recv = (float*) malloc((s_n + 1) * sizeof(float));
    float* tmp = (float*) malloc(local_n * sizeof(float));
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);


    boost::sort::spreadsort::float_sort(local_data, local_data + local_n);

    int r_odd, r_even;
    int n_odd, n_even;
    int even_rank = ((rank & 1) == 0);
    r_odd = even_rank ? rank - 1 : rank + 1;
    r_even = even_rank ? rank + 1 : rank - 1;

    n_even = r_even < remainder ? l_n : s_n;
    n_odd = r_odd < remainder ? l_n : s_n;
    
    if (r_odd >= size) r_odd = -1;
    if (r_even >= size) r_even = -1;

    float send_val, recv_val;
    bool need_exchange;
    
    for (int i = 0; i < size + 1; i += 2)
    {
        
        if (r_even != -1)
        {
            send_val = (rank < r_even) ? local_data[local_n - 1] : local_data[0];
            MPI_Sendrecv(
                &send_val, 1, MPI_FLOAT, r_even, 0,
                &recv_val, 1, MPI_FLOAT, r_even, 0,
                MPI_COMM_WORLD, &status
            );
            need_exchange = rank < r_even ? (send_val > recv_val) : (send_val < recv_val);
            if (need_exchange)
            {
                MPI_Sendrecv(
                    local_data, local_n, MPI_FLOAT, r_even, 0,
                    local_recv, n_even, MPI_FLOAT, r_even, 0,
                    MPI_COMM_WORLD, &status
                );
                if (rank < r_even) choose_min(local_data, local_recv, local_n, n_even, tmp);
                else choose_max(local_data, local_recv, local_n, n_even, tmp);
                std::swap(local_data, tmp);
            }
        }
        if (r_odd != -1)
        {
            send_val = (rank < r_odd) ? local_data[local_n - 1] : local_data[0];
            MPI_Sendrecv(
                &send_val, 1, MPI_FLOAT, r_odd, 0,
                &recv_val, 1, MPI_FLOAT, r_odd, 0,
                MPI_COMM_WORLD, &status
            );
            need_exchange = rank < r_odd ? (send_val > recv_val) : (send_val < recv_val);
            if (need_exchange)
            {
                MPI_Sendrecv(
                    local_data, local_n, MPI_FLOAT, r_odd, 0,
                    local_recv, n_odd, MPI_FLOAT, r_odd, 0,
                    MPI_COMM_WORLD, &status
                );
                if (rank < r_odd) choose_min(local_data, local_recv, local_n, n_odd, tmp);
                else choose_max(local_data, local_recv, local_n, n_odd, tmp);
                std::swap(local_data, tmp);
            }
        }
    }
    
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

