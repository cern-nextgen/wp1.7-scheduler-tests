#pragma once

#include "AlgorithmBase.hpp"

class ThirdAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit ThirdAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
};
