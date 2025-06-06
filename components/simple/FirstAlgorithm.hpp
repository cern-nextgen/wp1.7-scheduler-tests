#pragma once


#include "AlgorithmBase.hpp"


class FirstAlgorithm : public AlgorithmBase {
public:
    // Constructor with verbose and error parameters
    FirstAlgorithm(bool errorEnabled = false, int errorEventId = -1, bool verbose = false);

    StatusCode initialize() override;
    AlgCoInterface execute(EventContext ctx) const override;
    StatusCode finalize() override;

private:
    bool m_errorEnabled;  // Whether the error is enabled
    int m_errorEventId;   // Event ID where the error occurs
    bool m_verbose;       // Whether verbose output is enabled
};
