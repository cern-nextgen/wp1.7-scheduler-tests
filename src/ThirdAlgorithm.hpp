#pragma once


#include "AlgorithmBase.hpp"


class ThirdAlgorithm : public AlgorithmBase {
   static inline const std::vector<std::string> s_dependencies{"Object2"};
   static inline const std::vector<std::string> s_products{"Object4"};

public:
   virtual StatusCode initialize() override;
   virtual AlgCoInterface execute(EventContext ctx) const override;
   virtual StatusCode finalize() override;

   virtual const std::vector<std::string>& dependencies() const override;
   virtual const std::vector<std::string>& products() const override;
};
