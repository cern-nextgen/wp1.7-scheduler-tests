#pragma once


#include "AlgorithmBase.hpp"


class SecondAlgorithm : public AlgorithmBase {
public:
   virtual StatusCode initialize() override;
   virtual AlgCoInterface execute(EventContext ctx) const override;
   virtual StatusCode finalize() override;
};
