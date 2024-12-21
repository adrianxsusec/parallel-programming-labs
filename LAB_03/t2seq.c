#include <stdio.h>
#include <math.h>
#include <time.h>

double calculatePi(long n) {
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        double x = ((double)i - 0.5) / n;
        sum += 1.0 / (1.0 + x * x);
    }
    return 4.0 / n * sum;
}

int main() {
    long n = pow(10, 10);

    clock_t start = clock();
    double pi = calculatePi(n);
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    printf("PI = %.10f\n", pi);
    printf("Error: %.10f\n", pi - M_PI);
    printf("Time: %.3f\n", duration);

    return 0;
}