/// @file vector.c
/// @brief implementation of a dynamic array (vector) data structure
///
/// This file implements a generic vector data structure that can store elements of any type.
/// the vector automatically grows in size as elements are added. it supports operations like
/// push_back, get, concatenate, copy, shuffle, reverse etc.

#include "vector.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @brief initialize a vector with given element size and free function
/// @param vector pointer to vector to initialize
/// @param element_size size in bytes of each element
/// @param free_func destructor for individual elements
void vector_init_p(vector_t* vector, size_t element_size, free_func_t free_func) {
    vector->element_size = element_size;
    vector->size = 0;
    vector->capacity = 4;
    vector->data = malloc(vector->capacity * element_size);
    vector->free_func = free_func;
}

/// @brief create and return a new vector
/// @param element_size size in bytes of each element
/// @param free_func destructor for individual elements
/// @return initialized vector
vector_t vector_new_p(size_t element_size, free_func_t free_func) {
    vector_t vector;
    vector_init_p(&vector, element_size, free_func);
    return vector;
}

/// @brief free a vector and its elements
/// @param ptr pointer to vector to free
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

/// @brief reallocate vector memory to new capacity
/// @param vector pointer to vector to reallocate
/// @param new_capacity new capacity size
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

/// @brief add an element to the end of vector
/// @param vector pointer to vector
/// @param element pointer to element to add
void vector_push_back_p(vector_t* vector, const void* element) {
    if (vector->size == vector->capacity) {
        vector_realloc_p(vector, vector->capacity * 2);
    }
    memcpy((char*)vector->data + vector->size * vector->element_size, element,
           vector->element_size);
    vector->size++;
}

/// @brief get element at index in vector
/// @param vector pointer to vector
/// @param index index of element
/// @return pointer to element or NULL if index out of bounds
void* vector_get_p(const vector_t* vector, size_t index) {
    if (index < vector->size) {
        return (char*)vector->data + index * vector->element_size;
    }
    return NULL;
}

/// @brief concatenate src vector to end of dest vector
/// @param dest pointer to destination vector
/// @param src pointer to source vector to append
void vector_cat_p(vector_t* dest, const vector_t* src) {
    vector_realloc_p(dest, dest->size + src->size);
    memcpy((char*)dest->data + dest->size * dest->element_size, src->data,
           src->size * src->element_size);
    dest->size += src->size;
}

/// @brief copy src vector to dest vector
/// @param dest pointer to destination vector
/// @param src pointer to source vector to copy
void vector_copy_p(vector_t* dest, const vector_t* src) {
    vector_realloc_p(dest, src->size);
    memcpy(dest->data, src->data, src->size * src->element_size);
    dest->size = src->size;
}

/// @brief create a deep copy of a vector
/// @param src vector to clone
/// @return new vector containing copy of src
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

/// @brief check if vector contains element
/// @param vector pointer to vector to search
/// @param element pointer to element to find
/// @param element_size size of element in bytes
/// @return true if element found, false otherwise
bool vector_contains_p(const vector_t* vector, const void* element, size_t element_size) {
    for (size_t i = 0; i < vector->size; i++) {
        if (memcmp((char*)vector->data + i * vector->element_size, element, element_size) == 0) {
            return true;
        }
    }
    return false;
}

/// @brief serialize vector elements to string
/// @param dest destination buffer
/// @param size size of destination buffer
/// @param delim delimiter between elements
/// @param vector vector to serialize
/// @param element_serialize function to serialize each element
/// @return number of bytes written
int vector_serialize(char* dest, size_t size, const char* delim, vector_t vector,
                     int (*element_serialize)(char*, size_t, const void*)) {
    if (!dest || !size || !delim || !element_serialize) return 0;
    size_t offset = 0;
    size_t delim_len = strlen(delim);
    for (size_t i = 0; i < vector.size; i++) {
        if (offset >= size) return offset;
        int written = element_serialize(dest + offset, size - offset,
                                        (char*)vector.data + i * vector.element_size);
        if (written < 0 || (size_t)written >= size - offset) return offset;
        offset += written;
        if (i < vector.size - 1) {
            if (offset + delim_len >= size) return offset;
            memcpy(dest + offset, delim, delim_len);
            offset += delim_len;
        }
    }
    if (offset < size) {
        dest[offset] = '\0';
    }
    return offset;
}

/// @brief save vector to file
/// @param vector pointer to vector to save
/// @param file file handle to save to
void vector_save(const vector_t* vector, FILE* file) {
    fwrite(&vector->element_size, sizeof(vector->element_size), 1, file);
    fwrite(&vector->size, sizeof(vector->size), 1, file);
    fwrite(&vector->capacity, sizeof(vector->capacity), 1, file);
    fwrite(vector->data, vector->element_size, vector->size, file);
}

/// @brief load vector from file
/// @param vector pointer to vector to load into
/// @param file file handle to load from
void vector_load(vector_t* vector, FILE* file) {
    vector->free_func = NULL;
    fread(&vector->element_size, sizeof(vector->element_size), 1, file);
    fread(&vector->size, sizeof(vector->size), 1, file);
    fread(&vector->capacity, sizeof(vector->capacity), 1, file);
    vector->data = malloc(vector->capacity * vector->element_size);
    fread(vector->data, vector->element_size, vector->size, file);
}
