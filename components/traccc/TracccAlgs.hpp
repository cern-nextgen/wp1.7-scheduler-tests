#pragma once


#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <traccc/clusterization/clusterization_algorithm.hpp>
#include <traccc/finding/finding_algorithm.hpp>
#include <traccc/fitting/fitting_algorithm.hpp>
#include <traccc/geometry/detector.hpp>
#include <traccc/seeding/seeding_algorithm.hpp>
#include <traccc/seeding/spacepoint_formation_algorithm.hpp>
#include <traccc/seeding/track_params_estimation.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "AlgorithmBase.hpp"
#include "EventContext.hpp"


// Unique pointer members are moved insides execute. Not re-entrant.
class TracccCellsAlgorithm : public AlgorithmBase {
public:
   TracccCellsAlgorithm(int numEvents);
   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;

private:
   mutable vecmem::host_memory_resource m_mr;
   mutable std::unique_ptr<traccc::default_detector::host> m_detector;
   mutable std::unique_ptr<traccc::silicon_detector_description::host> m_det_descr;
   mutable std::unique_ptr<std::vector<traccc::edm::silicon_cell_collection::host>> m_cells;
   int m_numEvents;
};


class TracccComputeAlgorithm : public AlgorithmBase {
public:
   TracccComputeAlgorithm(int numEvents);
   StatusCode initialize() override;
   AlgCoInterface execute(EventContext ctx) const override;
   StatusCode finalize() override;

private:
   using spacepoint_formation_algorithm
       = traccc::host::spacepoint_formation_algorithm<traccc::default_detector::host>;

   using navigator_type = detray::navigator<const traccc::default_detector::host>;

   using stepper_type = detray::rk_stepper<detray::bfield::const_field_t::view_t,
                                           traccc::default_detector::host::algebra_type,
                                           detray::constrained_step<>>;

   using finding_algorithm = traccc::finding_algorithm<stepper_type, navigator_type>;

   using fitting_algorithm
       = traccc::fitting_algorithm<traccc::kalman_fitter<stepper_type, navigator_type>>;

private:
   mutable vecmem::host_memory_resource m_mr{};
   traccc::host::clusterization_algorithm m_clusterization{m_mr};
   spacepoint_formation_algorithm m_sf{m_mr};
   traccc::seeding_algorithm m_sa{traccc::seedfinder_config{},
                                  {traccc::seedfinder_config{}},
                                  traccc::seedfilter_config{},
                                  m_mr};
   traccc::track_params_estimation m_tp{m_mr};
   finding_algorithm m_finding_alg{finding_algorithm::config_type{}};
   fitting_algorithm m_fitting_alg{fitting_algorithm::config_type{}};
   int m_numEvents;
};
