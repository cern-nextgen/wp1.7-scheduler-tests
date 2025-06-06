#pragma once

#include "AlgorithmBase.hpp"

class SecondAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose parameter
    explicit SecondAlgorithm(bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_verbose; // Whether verbose output is enabled
};
