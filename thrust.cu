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

#define nx 400
#define ny 400
#define nz 400

#define ind(i, j, k) ((i) * ny * nz + (j) * nz + (k))

__global__ void average_along_I(double *A) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j > 0 && k > 0 && j < ny - 1 && k < nz - 1) {
    for (int i = 1; i < nx - 1; i++) {
      A[ind(i, j, k)] = (A[ind(i - 1, j, k)] + A[ind(i + 1, j, k)]) / 2;
    }
  }
}

__global__ void average_along_J(double *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && k > 0 && i < nx - 1 && k < nz - 1) {
    for (int j = 1; j < ny - 1; j++) {
      A[ind(i, j, k)] = (A[ind(i, j - 1, k)] + A[ind(i, j + 1, k)]) / 2;
    }
  }
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
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

void init(thrust::host_vector<double> &a);
void init(double (*a)[ny][nz]);
void compare(thrust::host_vector<double> &a1, double (*a2)[ny][nz]);

int main(int argc, char *argv[]) {
  enum ExecPath { Compare, Skip };

  const ExecPath Implementation = Skip;

  const double MAXEPS = 0.01;
  const int ITMAX = 100;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  std::chrono::steady_clock::time_point t1, t2;

  thrust::host_vector<double> A(nx * ny * nz);
  thrust::device_vector<double> A_device(nx * ny * nz);
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
    double eps = 0;

    double *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    // double *diff_device_ptr = thrust::raw_pointer_cast(diff_device.data());
    double *partly_reduce_ptr = thrust::raw_pointer_cast(partly_reduce.data());

    // std::cout << "calling I" << std::endl;

    t1 = std::chrono::steady_clock::now();
    average_along_I<<<dim3(ny / 32 + 1, nz / 32 + 1, 1), dim3(32, 32, 1)>>>(
        A_device_ptr);
    cudaDeviceSynchronize();
    t2 = std::chrono::steady_clock::now();
    std::cout
        << "I "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
        << "[ns]" << std::endl;

    if (Implementation == Compare) {
      for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
          for (int k = 1; k < nz - 1; k++)
            a[i][j][k] = (a[i - 1][j][k] + a[i + 1][j][k]) / 2;

      A = A_device;
      compare(A, a);
    }

    t1 = std::chrono::steady_clock::now();
    average_along_J<<<dim3(nx / 32 + 1, nz / 32 + 1, 1), dim3(32, 32, 1)>>>(
        A_device_ptr);
    cudaDeviceSynchronize();
    t2 = std::chrono::steady_clock::now();
    std::cout
        << "J "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
        << "[ns]" << std::endl;

    if (Implementation == Compare) {
      for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
          for (int k = 1; k < nz - 1; k++)
            a[i][j][k] = (a[i][j - 1][k] + a[i][j + 1][k]) / 2;

      A = A_device;
      compare(A, a);
    }

    t1 = std::chrono::steady_clock::now();
    average_along_K<<<dim3(nx / 32 + 1, ny / 32 + 1, 1), dim3(32, 32, 1)>>>(
        A_device_ptr, partly_reduce_ptr);
    cudaDeviceSynchronize();
    t2 = std::chrono::steady_clock::now();
    std::cout
        << "K"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
        << "[ns]" << std::endl;

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

    t1 = std::chrono::steady_clock::now();
    double eps_device =
        thrust::reduce(partly_reduce.begin(), partly_reduce.end(), 0.0,
                       thrust::maximum<double>());
    cudaDeviceSynchronize();
    t2 = std::chrono::steady_clock::now();
    std::cout
        << "reduction"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
        << "[ns]" << std::endl;

    printf(" IT = %4i   EPS = %14.7E\n, EPS_OLD =  %14.7E", it, eps_device,
           eps);
    if (eps_device < MAXEPS) break;
  }

  printf(" ADI Benchmark Completed.\n");
  printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
  printf(" Iterations      =       %12d\n", ITMAX);
  printf(" Operation type  =   double precision\n");

  printf(" END OF ADI Benchmark\n");

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  A = A_device;

  std::cout << "Size = " << nx  << "Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;

  

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