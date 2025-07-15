#include "TracccAlgs.hpp"

#include <iostream>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>

#include "EventStore.hpp"
#include "MemberFunctionName.hpp"


TracccCellsAlgorithm::TracccCellsAlgorithm(int numEvents)
    : m_mr{},
      m_detector{std::make_unique<traccc::default_detector::host>(m_mr)},
      m_det_descr{std::make_unique<traccc::silicon_detector_description::host>(m_mr)},
      m_cells{std::make_unique<std::vector<traccc::edm::silicon_cell_collection::host>>()},
      m_numEvents{numEvents} {
}


StatusCode TracccCellsAlgorithm::initialize() {
   SC_CHECK(addProduct<traccc::default_detector::host>("Detector"));
   SC_CHECK(addProduct<traccc::silicon_detector_description::host>(
       "DetectorDescription"));
   SC_CHECK(addProduct<std::vector<traccc::edm::silicon_cell_collection::host>>(
       "Cells"));

   traccc::io::read_detector_description(*m_det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");
   // Detector file, material file, grid file.
   traccc::io::read_detector(*m_detector,
                             m_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");

   m_cells->assign(m_numEvents, traccc::edm::silicon_cell_collection::host{m_mr});
   for(std::size_t i{}; auto& cells : *m_cells) {
      traccc::io::read_cells(cells, i++, "odd/geant4_10muon_10GeV/", &(*m_det_descr));
   }

   std::cout << MEMBER_FUNCTION_NAME(TracccCellsAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface TracccCellsAlgorithm::execute(AlgorithmContext ctx) const {
   SC_CHECK_YIELD(ctx.eventStore.record(std::move(m_detector), products()[0]));
   SC_CHECK_YIELD(ctx.eventStore.record(std::move(m_det_descr), products()[1]));
   SC_CHECK_YIELD(ctx.eventStore.record(std::move(m_cells), products()[2]));

   std::cout << MEMBER_FUNCTION_NAME(TracccCellsAlgorithm) << std::endl;
   co_return StatusCode::SUCCESS;
}


StatusCode TracccCellsAlgorithm::finalize() {
   std::cout << MEMBER_FUNCTION_NAME(TracccCellsAlgorithm) << std::endl;
   return StatusCode::SUCCESS;
}


TracccComputeAlgorithm::TracccComputeAlgorithm(int numEvents) : m_numEvents{numEvents} {
}


StatusCode TracccComputeAlgorithm::initialize() {
   SC_CHECK(addDependency<traccc::default_detector::host>("Detector"));
   SC_CHECK(addDependency<traccc::silicon_detector_description::host>("DetectorDescription"));
   SC_CHECK(addDependency<std::vector<traccc::edm::silicon_cell_collection::host>>("Cells"));
   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface TracccComputeAlgorithm::execute(AlgorithmContext ctx) const {
   const traccc::default_detector::host* detector_dep = nullptr;
   SC_CHECK_YIELD(ctx.eventStore.retrieve(detector_dep, dependencies()[0]));

   const traccc::silicon_detector_description::host* det_descr_dep = nullptr;
   SC_CHECK_YIELD(ctx.eventStore.retrieve(det_descr_dep, dependencies()[1]));

   const std::vector<traccc::edm::silicon_cell_collection::host>* cells_dep = nullptr;
   SC_CHECK_YIELD(ctx.eventStore.retrieve(cells_dep, dependencies()[2]));

   const auto& cells = (*cells_dep)[ctx.eventNumber % m_numEvents];
   std::cout << "Event number: " << ctx.eventNumber << std::endl;
   std::cout << "Size of cells: " << cells.size() << std::endl;

   auto measurements
       = m_clusterization(vecmem::get_data(cells), vecmem::get_data(*det_descr_dep));
   co_yield StatusCode::SUCCESS;

   auto spacepoints = m_sf(*detector_dep, vecmem::get_data(measurements));
   auto seeds = m_sa(spacepoints);
   co_yield StatusCode::SUCCESS;

   const traccc::vector3 field_vec{0.f, 0.f, traccc::seedfinder_config{}.bFieldInZ};
   const detray::bfield::const_field_t field{detray::bfield::create_const_field(field_vec)};

   auto params = m_tp(spacepoints, seeds, field_vec);
   auto track_candidates = m_finding_alg(*detector_dep, field, measurements, params);
   auto track_states = m_fitting_alg(*detector_dep, field, track_candidates);
   co_return StatusCode::SUCCESS;
}


StatusCode TracccComputeAlgorithm::finalize() {
   return StatusCode::SUCCESS;
}
