#pragma once


#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

#include "StatusCode.hpp"


struct ObjectHolderBase {
   ObjectHolderBase() = default;
   ObjectHolderBase(const ObjectHolderBase&) = delete;
   ObjectHolderBase& operator=(const ObjectHolderBase&) = delete;
   virtual ~ObjectHolderBase() = default;

   virtual void* get_pointer() const = 0;
   virtual const std::type_info& get_type() const = 0;
};


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


class EventStore {
public:
   template <typename T>
   bool contains(const std::string& name) const {
      auto it = m_dataStore.find(name);
      return (it != m_dataStore.end()) && (typeid(T) == it->second->get_type());
   }

   template <typename T>
   StatusCode retrieve(const T*& obj, const std::string& name) {
      assert(this->contains<T>(name));
      if(!this->contains<T>(name)) {
         return StatusCode::FAILURE;
      }

      obj = static_cast<const T*>(m_dataStore[name]->get_pointer());
      return StatusCode::SUCCESS;
   }

   template <typename T>
   StatusCode record(std::unique_ptr<T>&& obj, const std::string& name) {
      // Cannot record multiple objects with the same name.
      if(this->contains<T>(name)) {
         return StatusCode::FAILURE;
      }

      m_dataStore[name] = std::make_unique<ObjectHolder<T>>(std::move(obj));
      return StatusCode::SUCCESS;
   }

private:
   std::map<std::string, std::unique_ptr<ObjectHolderBase>> m_dataStore;
};

