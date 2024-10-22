#pragma once


#include "AlgorithmBase.hpp"


class FirstAlgorithm : public AlgorithmBase {
   static inline const std::vector<std::string> s_dependencies{};
   static inline const std::vector<std::string> s_products{"Object1", "Object2"};

public:
   virtual StatusCode initialize() override;
   virtual AlgCoInterface execute(EventContext ctx) const override;
   virtual StatusCode finalize() override;

   virtual const std::vector<std::string>& dependencies() const override;
   virtual const std::vector<std::string>& products() const override;
};
