#include <stdio.h>
#include <stdlib.h>

int sample_from_dist_d(const double* distribution, int size)
{
    double r = rand() / (double)RAND_MAX;
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += distribution[i];
        if (r < sum) {
            return i;
        }
    }
    return size - 1;
}

int sample_from_dist(const float* distribution, int size)
{
    double r = rand() / (double)RAND_MAX;
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += distribution[i];
        if (r < sum) {
            return i;
        }
    }
    return size - 1;
}

/*
int main() {
    int size = 3;
    double alpha[] = {1.0, 2.0, 3.0};
    double sample[3];

    dirichlet_distribution(alpha, sample, size);

    printf("Dirichlet sample: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", sample[i]);
    }
    printf("\n");

    return 0;
}
*/