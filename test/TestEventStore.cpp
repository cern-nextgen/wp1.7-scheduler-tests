// Mimic "assert always".
#ifdef NDEBUG
#undef NDEBUG
#endif


#define assert_not(expr) assert(not expr)


#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../src/EventStore.hpp"
#include "../src/StatusCode.hpp"


#define INITIALIZE_EVENTSTORE(T, x, name) \
   EventStore store;                      \
   assert(store.record(std::make_unique<T>(x), name))


template <typename T>
void run_contains(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   assert(store.contains<T>(name));
}


template <typename T>
void run_contains_fail(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   // Correct name, wrong type.
   assert_not(store.contains<T*>(name));
   assert_not(store.contains<T[]>(name));

   // Correct type, wrong name.
   assert_not(store.contains<T>(name + " "));
   assert_not(store.contains<T>(name + " "));
}


template <typename T>
void run_retrieve(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   const T* yptr = nullptr;
   assert(store.retrieve(yptr, name));
   assert(x == *yptr);
}


template <typename T>
void run_retrieve_fail(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   // Correct name, wrong type.
   const std::nullptr_t* null_val = nullptr;
   assert_not(store.retrieve(null_val, name));

   // Correct type, wrong name.
   const T* yptr = nullptr;
   assert_not(store.retrieve(yptr, name + " "));
}


// Make sure the same type with a different name can be recorded.
template <typename T>
void run_record_repeat(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   assert(store.record(std::make_unique<T>(x), name + " "));
}


template <typename T>
void run_record_repeat_fail(T x, const std::string& name) {
   INITIALIZE_EVENTSTORE(T, x, name);

   // Same type, same name is not allowed.
   assert_not(store.record(std::make_unique<T>(x), name));

   // Different type, same name is not allowed either.
   assert_not(store.record(std::make_unique<std::nullptr_t>(nullptr), name));
}


#define RUN_TEST(function_name, ...) \
   function_name(__VA_ARGS__);       \
   std::cout << "Passed " << #function_name << '.' << std::endl;


int main() {
   RUN_TEST(run_contains<int>, 1, "ObjectName");
   RUN_TEST(run_contains_fail<int>, 1, "ObjectName");

   RUN_TEST(run_retrieve<int>, 1, "ObjectName");
   RUN_TEST(run_retrieve_fail<int>, 1, "ObjectName");

   RUN_TEST(run_record_repeat<int>, 1, "ObjectName");
   RUN_TEST(run_record_repeat_fail<int>, 1, "ObjectName");

   return EXIT_SUCCESS;
}
