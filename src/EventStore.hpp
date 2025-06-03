#pragma once


#include <cassert>
#include <map>
#include <memory>
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
 * @brief Store for event data, allowing to record and retrieve objects by name. Objects can be any type. Type is also checked at retrieval time.
 * @note This class is not thread-safe.
 * @note The obecjt is returned by pointer.
 * @todo Get rid of pointer interface.
 */
class EventStore {
public:
   /**
    * @brief Test product presence in the store.
    * @tparam T product type
    * @param name Name of the product
    * @return boolean, whether the product of the right type is present.
    */
   template <typename T>
   bool contains(const std::string& name) const {
      auto it = m_dataStore.find(name);
      return (it != m_dataStore.end()) && (typeid(T) == it->second->get_type());
   }

   /**
    * @brief Get product
    * @tparam T product type
    * @param obj reference to pointer, ill be set to point to the product
    * @param name Name of the product
    * @return StatusCode (FAILURE if the product is not present or of the wrong type)
    */
   template <typename T>
   StatusCode retrieve(const T*& obj, const std::string& name) {
      if(!this->contains<T>(name)) {
         return StatusCode::FAILURE;
      }

      obj = static_cast<const T*>(m_dataStore[name]->get_pointer());
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
      // Cannot record multiple objects with the same name.
      if(m_dataStore.find(name) != m_dataStore.end()) {
         return StatusCode::FAILURE;
      }

      m_dataStore[name] = std::make_unique<ObjectHolder<T>>(std::move(obj));
      return StatusCode::SUCCESS;
   }

   /**
    * @brief Clears the store and deletes all the products.
    */
   void clear() {
      m_dataStore.clear();
   }

private:
   /// Map to hold the products by name.
   std::map<std::string, std::unique_ptr<ObjectHolderBase>> m_dataStore;
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
