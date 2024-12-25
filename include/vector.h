#ifndef VECTOR_H
#define VECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef void (*free_func_t)(void*);

typedef struct {
    void* data;
    size_t element_size;
    size_t size;
    size_t capacity;
    free_func_t free_func;
} vector_t;

vector_t vector_new_p(size_t element_size, free_func_t free_func);
void vector_free_p(void* ptr);
void vector_push_back_p(vector_t* vector, const void* element);
void* vector_get_p(const vector_t* vector, size_t index);
void vector_cat_p(vector_t* dest, const vector_t* src);
void vector_copy_p(vector_t* dest, const vector_t* src);
vector_t vector_clone(vector_t src);
void vector_shuffle(vector_t src);
void vector_reverse(vector_t src);
bool vector_contains_p(const vector_t* vector, const void* element, size_t element_size);
int vector_serialize(char* dest, size_t size, const char* delim, vector_t vector,
                     int (*element_serialize)(char*, size_t, const void*));
void vector_save(const vector_t* vector, FILE* file);
void vector_load(vector_t* vector, FILE* file);

#define vector_new(type, free_func)       vector_new_p(sizeof(type), free_func)
#define vector_init(type, vec)            vector_init_p(&(vec), sizeof(type))
#define vector_push_back(vec, value)      vector_push_back_p(&(vec), &(value))
#define vector_get(type, vec, index)      (*(type*)vector_get_p(&(vec), (index)))
#define vector_cat(dest, src)             vector_cat_p(&(dest), &(src))
#define vector_copy(dest, src)            vector_copy_p(&(dest), &(src))
#define vector_contains(type, vec, value) vector_contains_p(&(vec), &(value), sizeof(type))
#define vector_free(vec)                  vector_free_p(&(vec))
#define vector_data(type, vec)            ((type*)(vec).data)

#define for_each(type, vec, element)                                                             \
    for (type* element##_iter = (type*)(vec).data, element;                                      \
         (element##_iter < (type*)(vec).data + (vec).size) && (element = *element##_iter, true); \
         element##_iter++)

#define for_each_ptr(type, vec, ptr) \
    for (type* ptr = (type*)(vec).data; ptr < (type*)(vec).data + (vec).size; ptr++)

#endif