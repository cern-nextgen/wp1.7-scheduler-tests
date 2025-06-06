#pragma once


#include "AlgorithmBase.hpp"


class FirstAlgorithm : public AlgorithmBase {
public:
    // Constructor with optional error parameters
    FirstAlgorithm(bool errorEnabled = false, int errorEventId = -1);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_errorEnabled;  // Whether the error is enabled
    int m_errorEventId;   // Event ID where the error occurs
};
