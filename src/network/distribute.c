#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int sample_from_dist_d(const double* distribution, int size) {
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

int sample_from_dist(const float* distribution, int size) {
    double r = rand() / (double)RAND_MAX;
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += distribution[i];
        if (r <= sum) {
            return i;
        }
    }
    return size - 1;
}

void dist_normalize(float* dist, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += dist[i];
    }
    for (int i = 0; i < size; i++) {
        dist[i] /= sum;
    }
}

void dist_log(float* dist, int size) {
    for (int i = 0; i < size; i++) {
        dist[i] = log(dist[i] + 1e-8);
    }
}

void dist_softmax(float* dist, int size) {
    float max = -1e9;
    for (int i = 0; i < size; i++) {
        if (dist[i] > max) {
            max = dist[i];
        }
    }
    for (int i = 0; i < size; i++) {
        dist[i] = exp(dist[i] - max);
        // log_l("dist[i]: %f", dist[i]);
    }
    dist_normalize(dist, size);
}

void dist_set_temperature(float* dist, int size, float temp) {
    for (int i = 0; i < size; i++) {
        // log_l("%f -> %f", dist[i], dist[i] * 1.0 / temp);
        dist[i] *= 1.0 / temp;
    }
    // float sum = 0.0;
    // for (int i = 0; i < size; i++) {
    //     sum += dist[i];
    // }
    // for (int i = 0; i < size; i++) {
    //     dist[i] /= sum;
    // }
}

double rand_uniform() { return (double)rand() / RAND_MAX; }

double rand_normal() {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0;
}

double rand_gamma(double alpha) {
    if (alpha < 1.0) {
        double u = rand_uniform();
        return rand_gamma(alpha + 1.0) * pow(u, 1.0 / alpha);
    } else {
        double d = alpha - 1.0 / 3.0;
        double c = 1.0 / sqrt(9.0 * d);
        while (1) {
            double x = rand_normal();
            double v = 1.0 + c * x;
            if (v > 0) {
                v = v * v * v;
                double u = rand_uniform();
                if (u < 1.0 - 0.0331 * x * x * x || log(u) < 0.5 * x * x + d - v) {
                    return d * v;
                }
            }
        }
    }
}

void dirichlet_distribution(float* alpha, float* sample, int size) {
    double* gamma_samples = (double*)malloc(sizeof(double) * size);
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        gamma_samples[i] = rand_gamma(alpha[i]);
        sum += gamma_samples[i];
    }

    for (int i = 0; i < size; i++) {
        sample[i] = (float)(gamma_samples[i] / sum);
    }

    free(gamma_samples);
}

void simple_dirichlet_distribution(float alpha, float* sample, int size) {
    float* alpha_array = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        alpha_array[i] = alpha;
    }
    dirichlet_distribution(alpha_array, sample, size);
    free(alpha_array);
}

/*
int main() {
    srand(time(NULL));

    float select1_dist[] = {0, 0.9, 0.1};
    int cnt[] = {0, 0, 0};
    for (int i = 0; i < 1000; i++) {
        int x = sample_from_dist(select1_dist, 3);
        cnt[x]++;
    }
    printf("result: [%d, %d, %d]\n", cnt[0], cnt[1], cnt[2]);

    printf("Uniform sample: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", rand_uniform());
    }
    printf("\n");

    printf("Normal sample: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", rand_normal());
    }
    printf("\n");

    int size = 3;
    float alpha[] = {0.3, 0.3, 0.3};
    float sample[3];

    dirichlet_distribution(alpha, sample, size);

    printf("Dirichlet sample: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", sample[i]);
    }
    printf("\n");

    return 0;
}
*/