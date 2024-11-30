/* ADI program */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define nx 400
#define ny 400
#define nz 400

void init(double (*a)[ny][nz]);

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin =  std::chrono::steady_clock::now();

    double maxeps, eps;
    double (*a)[ny][nz];
    int it, itmax, i, j, k;
    double startt, endt;
    maxeps = 0.01;
    itmax = 100;
    a = (double (*)[ny][nz])malloc(nx * ny * nz * sizeof(double));
    init(a);

    for (it = 1; it <= itmax; it++)
    {
        eps = 0;        
        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a[i][j][k] = (a[i-1][j][k] + a[i+1][j][k]) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                    a[i][j][k] = (a[i][j-1][k] + a[i][j+1][k]) / 2;

        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++)
                {
                    double tmp1 = (a[i][j][k-1] + a[i][j][k+1]) / 2;
                    double tmp2 = fabs(a[i][j][k] - tmp1);
                    eps = Max(eps, tmp2);
                    a[i][j][k] = tmp1;
                }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }

    free(a);


  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Size = " << nx << " Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;


    return 0;
}

void init(double (*a)[ny][nz])
{
    int i, j, k;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[i][j][k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[i][j][k] = 0;
}
