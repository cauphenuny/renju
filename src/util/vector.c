#include "vector.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vector_init_p(vector_t* vector, size_t element_size, free_func_t free_func) {
    vector->element_size = element_size;
    vector->size = 0;
    vector->capacity = 4;
    vector->data = malloc(vector->capacity * element_size);
    vector->free_func = free_func;
}

vector_t vector_new_p(size_t element_size, free_func_t free_func) {
    vector_t vector;
    vector_init_p(&vector, element_size, free_func);
    return vector;
}

void vector_free_p(void* ptr) {
    vector_t* vec = ptr;
    if (!vec->capacity) return;
    if (vec->free_func) {
        for (size_t i = 0; i < vec->size; i++) {
            vec->free_func((char*)vec->data + i * vec->element_size);
        }
    }
    free(vec->data);
    vec->data = NULL;
    vec->free_func = NULL;
    vec->size = 0;
    vec->capacity = 0;
}

void vector_realloc_p(vector_t* vector, size_t new_capacity) {
    if (vector->capacity >= new_capacity) return;
    void* new_data = realloc(vector->data, new_capacity * vector->element_size);
    if (new_data == NULL) {
        fprintf(stderr, "failed to reallocate memory\n");
        exit(EXIT_FAILURE);
    }
    vector->data = new_data;
    vector->capacity = new_capacity;
}

void vector_push_back_p(vector_t* vector, const void* element) {
    if (vector->size == vector->capacity) {
        vector_realloc_p(vector, vector->capacity * 2);
    }
    memcpy((char*)vector->data + vector->size * vector->element_size, element,
           vector->element_size);
    vector->size++;
}

void* vector_get_p(const vector_t* vector, size_t index) {
    if (index < vector->size) {
        return (char*)vector->data + index * vector->element_size;
    }
    return NULL;
}

void vector_cat_p(vector_t* dest, const vector_t* src) {
    vector_realloc_p(dest, dest->size + src->size);
    memcpy((char*)dest->data + dest->size * dest->element_size, src->data,
           src->size * src->element_size);
    dest->size += src->size;
}

void vector_copy_p(vector_t* dest, const vector_t* src) {
    vector_realloc_p(dest, src->size);
    memcpy(dest->data, src->data, src->size * src->element_size);
    dest->size = src->size;
}

vector_t vector_clone(vector_t src) {
    vector_t dest = vector_new_p(src.element_size, src.free_func);
    vector_copy_p(&dest, &src);
    return dest;
}

void vector_shuffle(vector_t vector) {
    if (!vector.size) return;
    for (size_t i = vector.size - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        char* a = (char*)vector.data + i * vector.element_size;
        char* b = (char*)vector.data + j * vector.element_size;
        char temp[vector.element_size];
        memcpy(temp, a, vector.element_size);
        memcpy(a, b, vector.element_size);
        memcpy(b, temp, vector.element_size);
    }
}

void vector_reverse(vector_t vector) {
    for (size_t i = 0; i < vector.size / 2; i++) {
        char* a = (char*)vector.data + i * vector.element_size;
        char* b = (char*)vector.data + (vector.size - i - 1) * vector.element_size;
        char temp[vector.element_size];
        memcpy(temp, a, vector.element_size);
        memcpy(a, b, vector.element_size);
        memcpy(b, temp, vector.element_size);
    }
}

bool vector_contains_p(const vector_t* vector, const void* element, size_t element_size) {
    for (size_t i = 0; i < vector->size; i++) {
        if (memcmp((char*)vector->data + i * vector->element_size, element, element_size) == 0) {
            return true;
        }
    }
    return false;
}

int vector_serialize(char* dest, size_t size, const char* delim, vector_t vector,
                     int (*element_serialize)(char*, size_t, const void*)) {
    int offset = 0, delim_len = strlen(delim);
    for (size_t i = 0; i < vector.size; i++) {
        offset += element_serialize(dest + offset, size - offset,
                                    (char*)vector.data + i * vector.element_size);
        if (i < vector.size - 1) {
            strcat(dest + offset, delim);
            offset += delim_len;
        }
    }
    return offset;
}

void vector_save(const vector_t* vector, FILE* file) {
    fwrite(&vector->element_size, sizeof(vector->element_size), 1, file);
    fwrite(&vector->size, sizeof(vector->size), 1, file);
    fwrite(&vector->capacity, sizeof(vector->capacity), 1, file);
    fwrite(vector->data, vector->element_size, vector->size, file);
}

void vector_load(vector_t* vector, FILE* file) {
    vector->free_func = NULL;
    fread(&vector->element_size, sizeof(vector->element_size), 1, file);
    fread(&vector->size, sizeof(vector->size), 1, file);
    fread(&vector->capacity, sizeof(vector->capacity), 1, file);
    vector->data = malloc(vector->capacity * vector->element_size);
    fread(vector->data, vector->element_size, vector->size, file);
}

// int main()
// {
//     vector_t int_vector;
//     vector_init(int, int_vector);
//
//     int value = 10;
//     vector_push_back(int_vector, value);
//     value = 20;
//     vector_push_back(int_vector, value);
//
//     int retrieved_value = vector_get(int, int_vector, 0);
//     printf("First element: %d\n", retrieved_value);
//     retrieved_value = vector_get(int, int_vector, 1);
//     printf("Second element: %d\n", retrieved_value);
//
//     vector_free(&int_vector);
//     return 0;
// }