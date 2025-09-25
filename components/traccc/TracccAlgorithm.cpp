#include "TracccAlgorithm.hpp"

#include <iostream>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>


TracccAlgorithm::TracccAlgorithm(int numEvents) : m_numEvents{numEvents} {
}


StatusCode TracccAlgorithm::initialize() {
   traccc::io::read_detector_description(m_det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");
   // Detector file, material file, grid file.
   traccc::io::read_detector(m_detector,
                             m_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");

   m_cells.assign(m_numEvents, traccc::edm::silicon_cell_collection::host{m_mr});
   for(std::size_t i{}; auto& cells : m_cells) {
      traccc::io::read_cells(cells, i++, "odd/geant4_10muon_10GeV/", &m_det_descr);
   }

   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface TracccAlgorithm::execute(AlgorithmContext ctx) const {
   const auto& cells = m_cells[ctx.eventNumber % m_numEvents];
   std::cout << "Event number: " << ctx.eventNumber << std::endl;
   std::cout << "Size of cells: " << cells.size() << std::endl;

   auto measurements
       = m_clusterization(vecmem::get_data(cells), vecmem::get_data(m_det_descr));
   co_yield StatusCode::SUCCESS;

   auto spacepoints = m_sf(m_detector, vecmem::get_data(measurements));
   auto seeds = m_sa(spacepoints);
   co_yield StatusCode::SUCCESS;

   const traccc::vector3 field_vec{0.f, 0.f, traccc::seedfinder_config{}.bFieldInZ};
   const detray::bfield::const_field_t field{detray::bfield::create_const_field(field_vec)};

   auto params = m_tp(spacepoints, seeds, field_vec);
   auto track_candidates = m_finding_alg(m_detector, field, measurements, params);
   auto track_states = m_fitting_alg(m_detector, field, track_candidates);
   co_return StatusCode::SUCCESS;
}


StatusCode TracccAlgorithm::finalize() {
   return StatusCode::SUCCESS;
}
