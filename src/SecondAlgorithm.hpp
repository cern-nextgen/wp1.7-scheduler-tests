#pragma once


#include "AlgorithmBase.hpp"


class SecondAlgorithm : public AlgorithmBase {
   static inline const std::vector<std::string> s_dependencies{"Object1"};
   static inline const std::vector<std::string> s_products{"Object3"};

public:
   virtual StatusCode initialize() override;
   virtual AlgCoInterface execute(EventContext ctx) const override;
   virtual StatusCode finalize() override;

   virtual const std::vector<std::string>& dependencies() const override;
   virtual const std::vector<std::string>& products() const override;
};
