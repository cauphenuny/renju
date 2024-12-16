#ifndef VECTOR_H
#define VECTOR_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    void* data;
    size_t element_size;
    size_t size;
    size_t capacity;
} vector_t;

void vector_init_impl(vector_t* vector, size_t element_size);
vector_t vector_new_impl(size_t element_size);
void vector_push_back_impl(vector_t* vector, void* element);
void* vector_get_impl(vector_t* vector, size_t index);
void vector_free_impl(vector_t* ptr);
void vector_cat_impl(vector_t* dest, vector_t* src);
bool vector_contains_impl(vector_t* vector, void* element, size_t element_size);

#define vector_new(type)                  vector_new_impl(sizeof(type))
#define vector_init(type, vec)            vector_init_impl(&(vec), sizeof(type))
#define vector_free(vec)                  vector_free_impl(&(vec))
#define vector_push_back(vec, value)      vector_push_back_impl(&(vec), &(value))
#define vector_get(type, vec, index)      (*(type*)vector_get_impl(&(vec), (index)))
#define vector_cat(dest, src)             vector_cat_impl(&(dest), &(src))
#define vector_contains(type, vec, value) vector_contains_impl(&(vec), &(value), sizeof(type))

#define for_all_elements(type, vec, element)                                  \
    for (type* element##_iter = (type*)(vec).data, element = *element##_iter; \
         element##_iter < (type*)(vec).data + (vec).size;                     \
         element##_iter++, element = *element##_iter)

#endif