#ifndef DISTRIBUTE_H
#define DISTRIBUTE_H

int sample_from_dist_d(const double* distribution, int size);
int sample_from_dist(const float* distribution, int size);
void dist_normalize(float* dist, int size);
void dist_log(float* dist, int size);
void dist_softmax(float* dist, int size);
void dist_set_temperature(float* dist, int size, float temp);
void dirichlet_distribution(float* alpha, float* sample, int size);
void simple_dirichlet_distribution(float alpha, float* sample, int size);

#endif
