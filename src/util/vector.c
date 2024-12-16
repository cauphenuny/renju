#include "vector.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vector_init_impl(vector_t* vector, size_t element_size) {
    vector->element_size = element_size;
    vector->size = 0;
    vector->capacity = 4;
    vector->data = malloc(vector->capacity * element_size);
}

vector_t vector_new_impl(size_t element_size) {
    vector_t vector;
    vector_init_impl(&vector, element_size);
    return vector;
}

void vector_realloc_impl(vector_t* vector, size_t new_capacity) {
    if (vector->capacity >= new_capacity) return;
    vector->data = realloc(vector->data, new_capacity * vector->element_size);
    vector->capacity = new_capacity;
}

void vector_push_back_impl(vector_t* vector, void* element) {
    if (vector->size == vector->capacity) {
        vector->capacity *= 2;
        vector->data = realloc(vector->data, vector->capacity * vector->element_size);
    }
    memcpy((char*)vector->data + vector->size * vector->element_size, element,
           vector->element_size);
    vector->size++;
}

void* vector_get_impl(vector_t* vector, size_t index) {
    if (index < vector->size) {
        return (char*)vector->data + index * vector->element_size;
    }
    return NULL;
}

void vector_cat_impl(vector_t* dest, vector_t* src) {
    vector_realloc_impl(dest, dest->size + src->size);
    memcpy((char*)dest->data + dest->size * dest->element_size, src->data,
           src->size * src->element_size);
    dest->size += src->size;
}

void vector_free_impl(vector_t* ptr) {
    free(ptr->data);
    ptr->data = NULL;
    ptr->size = 0;
    ptr->capacity = 0;
}

bool vector_contains_impl(vector_t* vector, void* element, size_t element_size) {
    for (size_t i = 0; i < vector->size; i++) {
        if (memcmp((char*)vector->data + i * vector->element_size, element, element_size) == 0) {
            return true;
        }
    }
    return false;
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