#pragma once


#include "AlgorithmBase.hpp"


class ThirdAlgorithm : public AlgorithmBase {
public:
   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;
};
