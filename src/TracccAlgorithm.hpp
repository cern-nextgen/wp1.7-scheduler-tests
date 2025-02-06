#pragma once


#include <traccc/clusterization/clusterization_algorithm.hpp>
#include <traccc/geometry/detector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "AlgorithmBase.hpp"
#include "EventContext.hpp"


class TracccAlgorithm : public AlgorithmBase {
public:
   TracccAlgorithm(int numEvents);
   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;

private:
   mutable vecmem::host_memory_resource m_mr;
   traccc::host::clusterization_algorithm m_clusterization{m_mr};
   traccc::silicon_detector_description::host m_det_descr{m_mr};
   traccc::default_detector::host m_detector{m_mr};
   std::vector<traccc::edm::silicon_cell_collection::host> m_cells;
   int m_numEvents;
};
