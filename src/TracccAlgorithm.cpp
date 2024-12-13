// Local include(s).
#include "TracccAlgorithm.hpp"

#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <iostream>
#include <traccc/finding/finding_algorithm.hpp>
#include <traccc/fitting/fitting_algorithm.hpp>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>
#include <traccc/seeding/seeding_algorithm.hpp>
#include <traccc/seeding/spacepoint_formation_algorithm.hpp>
#include <traccc/seeding/track_params_estimation.hpp>

StatusCode TracccAlgorithm::initialize() {
   traccc::io::read_detector_description(m_det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");
   // detector file, material file, grid file
   traccc::io::read_detector(m_detector,
                             m_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");


   return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface TracccAlgorithm::execute(EventContext ctx) const {
   //  seeding algorithm
   traccc::edm::silicon_cell_collection::host cells{m_mr};

   traccc::io::read_cells(cells, ctx.eventNumber, "odd/geant4_10muon_10GeV/", &m_det_descr);
   std::cout << "Event number: " << ctx.eventNumber << std::endl;
   std::cout << "Size of cells: " << cells.size() << std::endl;

   auto measurements
       = m_clusterization(vecmem::get_data(cells), vecmem::get_data(m_det_descr));
   std::cout << "Size of measurements: " << measurements.size() << std::endl;
   co_yield StatusCode::SUCCESS;

   using spacepoint_formation_algorithm
       = traccc::host::spacepoint_formation_algorithm<traccc::default_detector::host>;

   spacepoint_formation_algorithm sf{m_mr};
   spacepoint_formation_algorithm::output_type spacepoints{&m_mr};
   spacepoints = sf(m_detector, vecmem::get_data(measurements));
   std::cout << "Size of spacepoints: " << spacepoints.size() << std::endl;

   traccc::seeding_algorithm::output_type seeds{&m_mr};
   traccc::seeding_algorithm sa{traccc::seedfinder_config{},
                                {traccc::seedfinder_config{}},
                                traccc::seedfilter_config{},
                                m_mr};
   seeds = sa(spacepoints);
   std::cout << "Size of seeds" << seeds.size() << std::endl;
   co_yield StatusCode::SUCCESS;

   using navigator_type = detray::navigator<const traccc::default_detector::host>;
   using stepper_type = detray::rk_stepper<detray::bfield::const_field_t::view_t,
                                           traccc::default_detector::host::algebra_type,
                                           detray::constrained_step<>>;
   using finding_algorithm = traccc::finding_algorithm<stepper_type, navigator_type>;

   finding_algorithm finding_alg{finding_algorithm::config_type{}};
   finding_algorithm::output_type track_candidates{&m_mr};

   const traccc::vector3 field_vec = {0.f, 0.f, traccc::seedfinder_config{}.bFieldInZ};
   const detray::bfield::const_field_t field = detray::bfield::create_const_field(field_vec);

   traccc::track_params_estimation::output_type params{&m_mr};

   track_candidates = finding_alg(m_detector, field, measurements, params);

   std::cout << "Size of track_candidates: " << track_candidates.size() << std::endl;

   using fitting_algorithm
       = traccc::fitting_algorithm<traccc::kalman_fitter<stepper_type, navigator_type>>;
   fitting_algorithm fitting_alg{fitting_algorithm::config_type{}};
   fitting_algorithm::output_type track_states{&m_mr};
   track_states = fitting_alg(m_detector, field, track_candidates);

   co_return StatusCode::SUCCESS;
}

StatusCode TracccAlgorithm::finalize() {
   return StatusCode::SUCCESS;
}

const std::vector<std::string> &TracccAlgorithm::dependencies() const {
   static std::vector<std::string> deps = {};
   return deps;
}

const std::vector<std::string> &TracccAlgorithm::products() const {
   static std::vector<std::string> prods = {};
   return prods;
}
