/* ADI program */
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define nx 1024
#define ny 1024
#define nz 1024

#define IsInBounds(x,a,b) ((x >= a) && (x <= b))
#define ind(i, j, k) ((i) * ny * nz + (j) * nz + (k))

/*__device__ __host__ int ind(int i, int j, int k){
  return ((i) * ny * nz + (j) * nz + (k));
}*/

class Timer {
 public:
  Timer(std::string message) : message(message) {
    t1 = std::chrono::steady_clock::now();
  }

  ~Timer() {
    t2 = std::chrono::steady_clock::now();

    /*std::cout << "section : " << message << " took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "[ms]" << std::endl;*/
  }

  std::string message;
  std::chrono::steady_clock::time_point t1, t2;
};

__global__ void average_along_I(double *A) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (j > 0 && k > 0 && j < ny - 1 && k < nz - 1) {
    for (int i = 1; i < nx - 1; i++) {
      A[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i + 1, j, k)]) / 2;
    }
  }
}

__global__ void average_along_J(double *A) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && k > 0 && i < nx - 1 && k < nz - 1) {
    for (int j = 1; j < ny - 1; j++) {
      A[ind(i, j, k)] = (A[ind(i, j - 1, k)] + A[ind(i, j + 1, k)]) / 2;
    }
  }
}







const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

//transpose (j,k) components
__global__ void transpose(double *A) {
  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i < nx && k < nz ) {
    for (int j = 1; j < k; j++) {
        thrust::swap(A[ind(i, j, k)], A[ind(i, k, j)]);
    }
  }
}

__global__ void transposeCoalesced(const double *idata, double *odata)
{
  __shared__ double tile[TILE_DIM][TILE_DIM];


  //Calculating offsets to the beginning of the tile
  int k = blockIdx.x * TILE_DIM + threadIdx.x;
  int j = blockIdx.y;
  int i = blockIdx.z * TILE_DIM + threadIdx.y;

  /* 1111
     2222
     1111
     2222 */
  for (int q = 0; q < TILE_DIM; q += BLOCK_ROWS){
     int index  = ind(i+q, j, k);
     tile[threadIdx.y + q][  threadIdx.x] = idata[index];
  }
  __syncthreads();


  //Offset to the tile in the output array
  k = blockIdx.z * TILE_DIM + threadIdx.x;  
  i = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int q = 0; q < TILE_DIM; q += BLOCK_ROWS){    
     odata[ind(i + q, j, k)] = tile[threadIdx.x][threadIdx.y + q];
    }
}


void call_transpose(double* A_device_ptr, double * A_device_T_ptr){
      transposeCoalesced<<<dim3(nz / 32, ny, nz / 32), dim3(32, BLOCK_ROWS, 1)>>>(
        A_device_ptr, A_device_T_ptr);

}



__global__ void average_along_K(double *A, double *partly_reduce) {
  __shared__ double sdata[1024];  // block size

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  double eps = 0;

  if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
    for (int k = 1; k < nz - 1; k++) {
      double tmp1 = (A[ind(i, j, k - 1)] + A[ind(i, j, k + 1)]) / 2;
      double tmp2 = fabs(A[ind(i, j, k)] - tmp1);
      eps = Max(eps, tmp2);
      A[ind(i, j, k)] = tmp1;
    }

    // partly_reduce[i * ny + j] = eps;
    sdata[threadIdx.x * blockDim.y + threadIdx.y] = eps;
  } else {
    sdata[threadIdx.x * blockDim.y + threadIdx.y] = 0;
  }

  int tid = threadIdx.x * blockDim.y + threadIdx.y;

  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x * blockDim.y) {
      sdata[index] = Max(sdata[index], sdata[index + s]);
    }
    __syncthreads();
  }

  if (tid == 0) partly_reduce[blockIdx.x * gridDim.y + blockIdx.y] = sdata[0];
}

template <typename T>
struct absolute_value : public unary_function<T, T> {
  __host__ __device__ T operator()(const  T &x) const  {
    return x < T(0) ? -x : x;
  }
};

void init(thrust::host_vector<double> &a);
void init(double (*a)[ny][nz]);
void compare(thrust::host_vector<double> &a1, double (*a2)[ny][nz]);

int main(int argc, char *argv[]) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  enum ExecPath { Compare, Skip };

  const  ExecPath Implementation = Skip;

  const  double MAXEPS = 0.0001;
  const  int ITMAX = 100;



  thrust::host_vector<double> A(nx * ny * nz);
  thrust::device_vector<double> A_device(nx * ny * nz), A_device2(nx * ny * nz), eps_device(nx * ny * nz);

  thrust::device_vector<double> A_device_T(nx * ny * nz);

  // thrust::device_vector<double> diff_device(nx * ny * nz, 0);
  thrust::device_vector<double> partly_reduce((nx / 32 + 1) * (ny / 32 + 1), 0);

  init(A);

  A_device = A;

  double(*a)[ny][nz] = nullptr;

  if (Implementation == Compare) {
    a = (double(*)[ny][nz])malloc(nx * ny * nz * sizeof(double));
    init(a);
  }

  for (int it = 1; it <= ITMAX; it++) {
    std::cout << "it = " << it << std::endl;

    Timer t("Cycle total");

    double eps = 0;

    double *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    // double *diff_device_ptr = thrust::raw_pointer_cast(diff_device.data());
    double *partly_reduce_ptr = thrust::raw_pointer_cast(partly_reduce.data());

    // std::cout << "calling I" << std::endl;

    {
      Timer t("I");

      average_along_I<<<dim3(nz / 32 + 1, ny / 32 + 1, 1), dim3(32, 32, 1)>>>(
          A_device_ptr);
      cudaDeviceSynchronize();
    }

    if (Implementation == Compare) {
      for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
          for (int k = 1; k < nz - 1; k++)
            a[i][j][k] = (a[i - 1][j][k] + a[i + 1][j][k]) / 2;

      A = A_device;
      compare(A, a);
    }

    {
    Timer t("J");
    average_along_J<<<dim3(nz / 32 + 1, nx / 32 + 1, 1), dim3(32, 32, 1)>>>(
        A_device_ptr);
    cudaDeviceSynchronize();
   }

    if (Implementation == Compare) {
      for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
          for (int k = 1; k < nz - 1; k++)
            a[i][j][k] = (a[i][j - 1][k] + a[i][j + 1][k]) / 2;

      A = A_device;
      compare(A, a);
    }





    double *A_device_T_ptr = thrust::raw_pointer_cast(A_device_T.data());
    double *A_device2_ptr = thrust::raw_pointer_cast(A_device2.data());

    // Transposing the matrix
    {
      Timer t("transpose");
      call_transpose(A_device_ptr, A_device_T_ptr);
      cudaDeviceSynchronize();
    }

    {
      Timer t("I");
      average_along_I<<<dim3(nz / 32 + 1, ny / 32 + 1, 1), dim3(32, 32, 1)>>>(
          A_device_T_ptr);
      cudaDeviceSynchronize();
    }

    {
      Timer t("transpose");
      call_transpose(A_device_T_ptr, A_device2_ptr);
      cudaDeviceSynchronize();
    }

  
  {
      Timer t("minus");
    thrust::transform(A_device.begin(), A_device.end(), A_device2.begin(),
                      eps_device.begin(), thrust::minus<double>());
  }


  double eps_val = 0;
  {
      Timer t("reduce");
      eps_val =        thrust::transform_reduce(eps_device.begin(), eps_device.end(),  absolute_value<double>(), 0.0,
                       thrust::maximum<double>());
  }

    {
    Timer t("copy");
    A_device = A_device2;
    }

    if (Implementation == Compare) {
      for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
          for (int k = 1; k < nz - 1; k++) {
            double tmp1 = (a[i][j][k - 1] + a[i][j][k + 1]) / 2;
            double tmp2 = fabs(a[i][j][k] - tmp1);
            eps = Max(eps, tmp2);
            a[i][j][k] = tmp1;
          }

      A = A_device;
      compare(A, a);
    }



    printf(" IT = %4i   EPS = %14.7E\n, EPS_OLD =  %14.7E", it, eps_val,     eps);

    if (eps_val < MAXEPS) 
      break;

  //return 0;
  //break;
  }

  printf(" ADI Benchmark Completed.\n");
  printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
  printf(" Iteratioms      =       %12d\n", ITMAX);
  printf(" Operation type  =   double precision\n");

  printf(" END OF ADI Benchmark\n");

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Size = " << nx << " Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -    begin)
                   .count()
            << "[ms]" << std::endl;


  A = A_device;

  std::cout << "Size = " << nx << endl; 
  

  if (Implementation == Compare) {
    std::ofstream dump_file(std::string(argv[0]) + ".dump", std::ios::trunc);
    if (dump_file.is_open()) {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++) dump_file << A[ind(i, j, k)];
    }
    dump_file.close();
  }
}

void init(thrust::host_vector<double> &a) {
  int i, j, k;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
        if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 ||
            i == nx - 1)
          a[ind(i, j, k)] =
              10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
        else
          a[ind(i, j, k)] = 0;
}

void init(double (*a)[ny][nz]) {
  int i, j, k;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
        if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 ||
            i == nx - 1)
          a[i][j][k] =
              10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
        else
          a[i][j][k] = 0;
}

void compare(thrust::host_vector<double> &a1, double (*a2)[ny][nz]) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        if (a1[ind(i, j, k)] != a2[i][j][k])
          throw std::runtime_error("Error occured");
      }
}