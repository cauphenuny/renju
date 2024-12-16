#ifndef VECTOR_H
#define VECTOR_H

#include <stdbool.h>
#include <stddef.h>

typedef void (*free_func_t)(void*);

typedef struct {
    void* data;
    size_t element_size;
    size_t size;
    size_t capacity;
    free_func_t free_func;
} vector_t;

vector_t vector_new_impl(size_t element_size, free_func_t free_func);
void vector_free(void* ptr);
void vector_push_back_impl(vector_t* vector, void* element);
void* vector_get_impl(vector_t* vector, size_t index);
void vector_cat_impl(vector_t* dest, vector_t* src);
void vector_copy_impl(vector_t* dest, vector_t* src);
vector_t vector_clone_impl(vector_t* src);
bool vector_contains_impl(vector_t* vector, void* element, size_t element_size);

#define vector_new(type, free_func)       vector_new_impl(sizeof(type), free_func)
#define vector_init(type, vec)            vector_init_impl(&(vec), sizeof(type))
#define vector_push_back(vec, value)      vector_push_back_impl(&(vec), &(value))
#define vector_get(type, vec, index)      (*(type*)vector_get_impl(&(vec), (index)))
#define vector_cat(dest, src)             vector_cat_impl(&(dest), &(src))
#define vector_copy(dest, src)            vector_copy_impl(&(dest), &(src))
#define vector_clone(src)                 vector_clone_impl(&(src))
#define vector_contains(type, vec, value) vector_contains_impl(&(vec), &(value), sizeof(type))

#define for_each(type, vec, element)                                                             \
    for (type* element##_iter = (type*)(vec).data, element;                                      \
         (element##_iter < (type*)(vec).data + (vec).size) && (element = *element##_iter, true); \
         element##_iter++)

#define for_each_ptr(type, vec, ptr) \
    for (type* ptr = (type*)(vec).data; ptr < (type*)(vec).data + (vec).size; ptr++)

#endif