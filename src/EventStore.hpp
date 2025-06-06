#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "EventContext.hpp"
#include "StatusCode.hpp"

/**
 * @brief And `std::any`-style container for unique pointer (abstract base class).
 */
struct ObjectHolderBase {
   ObjectHolderBase() = default;
   virtual ~ObjectHolderBase() = default;

   virtual void* get_pointer() const = 0;
   virtual const std::type_info& get_type() const = 0;
};

/**
 * @brief Concrete implementation of `ObjectHolderBase` for object type `T`.
 * @tparam `T` the type of the object pointers to hold.
 */
template <typename T>
class ObjectHolder : public ObjectHolderBase {
public:
   ObjectHolder(std::unique_ptr<T>&& ptr) : m_ptr{std::move(ptr)} {
      assert(m_ptr);
   }

   void* get_pointer() const override {
      return m_ptr.get();
   }

   const std::type_info& get_type() const override {
      return typeid(*m_ptr);
   }

private:
   std::unique_ptr<T> m_ptr;
};

/**
 * @brief Struct to hold the value and its associated mutex.
 */
struct ValueWithMutex {
   std::unique_ptr<ObjectHolderBase> value;
   mutable std::shared_mutex mutex;

   // Default constructor
   ValueWithMutex() = default;

   // Move constructor
   ValueWithMutex(ValueWithMutex&& other) noexcept
       : value(std::move(other.value)) {
       // Note: `mutex` is default-constructed for the new object
   }

   // Move assignment operator
   ValueWithMutex& operator=(ValueWithMutex&& other) noexcept {
       if (this != &other) {
           value = std::move(other.value);
           // Note: `mutex` is not moved; it remains default-constructed
       }
       return *this;
   }

   // Deleted copy constructor and copy assignment operator
   ValueWithMutex(const ValueWithMutex&) = delete;
   ValueWithMutex& operator=(const ValueWithMutex&) = delete;
};

/**
 * @brief Store for event data, allowing to record and retrieve objects by name. Objects can be any type. Type is also checked at retrieval time.
 */
class EventStore {
public:
   using KeyType = std::string;
   using ValueType = ValueWithMutex;

   // Trivial constructor
   EventStore() = default;

   // Move constructor
   EventStore(EventStore&& other) noexcept {
      std::unique_lock<std::shared_mutex> lock(other.m_globalMutex); // Lock the source's global mutex
      m_store = std::move(other.m_store);
   }

   // Move assignment operator
   EventStore& operator=(EventStore&& other) noexcept {
      if (this != &other) {
         std::unique_lock<std::shared_mutex> lockThis(m_globalMutex, std::defer_lock);
         std::unique_lock<std::shared_mutex> lockOther(other.m_globalMutex, std::defer_lock);
         std::lock(lockThis, lockOther); // Lock both mutexes
         m_store = std::move(other.m_store);
      }
      return *this;
   }

   /**
    * @brief Test product presence in the store.
    * @tparam T product type
    * @param name Name of the product
    * @return boolean, whether the product of the right type is present.
    */
   template <typename T>
   bool contains(const std::string& name) const {
      std::shared_lock<std::shared_mutex> globalLock(m_globalMutex); // Shared lock for map access
      auto it = m_store.find(name);
      if (it == m_store.end()) {
         return false;
      }
      std::shared_lock<std::shared_mutex> lock(it->second.mutex); // Lock the element mutex
      return typeid(T) == it->second.value->get_type();
   }

   /**
    * @brief Get product
    * @tparam T product type
    * @param obj reference to pointer, will be set to point to the product
    * @param name Name of the product
    * @return StatusCode (FAILURE if the product is not present or of the wrong type)
    */
   template <typename T>
   StatusCode retrieve(const T*& obj, const std::string& name) {
      std::shared_lock<std::shared_mutex> globalLock(m_globalMutex); // Shared lock for map access
      auto it = m_store.find(name);
      if (it == m_store.end()) {
         return StatusCode::FAILURE;
      }
      std::shared_lock<std::shared_mutex> lock(it->second.mutex); // Lock the element mutex
      if (typeid(T) != it->second.value->get_type()) {
         return StatusCode::FAILURE;
      }
      obj = static_cast<const T*>(it->second.value->get_pointer());
      return StatusCode::SUCCESS;
   }

   /**
    * @brief Record a product in the store.
    * @tparam T product type
    * @param obj Unique pointer to the product to be recorded. Ownership is transferred to the store.
    * @param name Name of the product.
    * @return StatusCode (FAILURE if the product with the same name is already present)
    */
   template <typename T>
   StatusCode record(std::unique_ptr<T>&& obj, const std::string& name) {
      std::unique_lock<std::shared_mutex> globalLock(m_globalMutex); // Exclusive lock for map modification
      auto it = m_store.find(name);
      if (it != m_store.end()) {
         return StatusCode::FAILURE;
      }

      ValueWithMutex newValue;
      newValue.value = std::make_unique<ObjectHolder<T>>(std::move(obj));
      m_store[name] = std::move(newValue);
      return StatusCode::SUCCESS;
   }

   /**
    * @brief Clears the store and deletes all the products.
    */
   void clear() {
      std::unique_lock<std::shared_mutex> globalLock(m_globalMutex); // Exclusive lock for map modification
      m_store.clear();
   }

private:
   mutable std::shared_mutex m_globalMutex; // Global lock for the map
   std::map<KeyType, ValueType> m_store;    // The map storing the elements with their associated mutexes
};

/**
 * @brief Singleton registry for event stores, indexed by slot number.
 * Initialized in Scheduler::initialize(). Then referenced by `eventStoreOf()` function.
 */
class EventStoreRegistry {
// Private first to allow static functions to access it.
private:
   EventStoreRegistry() = default;

   /// Singleton instance handler
   static std::vector<EventStore>& gInstance() {
      static std::vector<EventStore> eventStores;
      return eventStores;
   }
public:
   EventStoreRegistry(const EventStoreRegistry&) = delete;
   EventStoreRegistry& operator=(const EventStoreRegistry&) = delete;

   /**
    * @brief Singleton accessor and instance creator/holder.
    * @return reference to the singleton instance of EventStoreRegistry.
    */
   static EventStore& of(const EventContext& ctx) {
      return gInstance().at(ctx.slotNumber);
   }

   static void initialize(std::size_t slots) {
      gInstance().resize(slots);
   }
   
};
