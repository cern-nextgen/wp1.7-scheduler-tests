// Local include(s).
#include "TracccAlgorithm.hpp"

#include <iostream>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>
#include <traccc/seeding/spacepoint_formation_algorithm.hpp>

StatusCode TracccAlgorithm::initialize() {
   return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface TracccAlgorithm::execute(EventContext ctx) const {
   //  add spacepoint formation and a seeding algorithm
   traccc::edm::silicon_cell_collection::host cells{m_mr};
   traccc::silicon_detector_description::host det_descr{m_mr};

   traccc::io::read_detector_description(det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");

   traccc::io::read_cells(cells, ctx.eventNumber, "odd/geant4_10muon_10GeV/", &det_descr);
   std::cout << "Event number: " << ctx.eventNumber << std::endl;
   std::cout << "Number of cells: " << cells.size() << std::endl;

   auto measurements = m_clusterization(vecmem::get_data(cells), vecmem::get_data(det_descr));
   std::cout << "Number of measurements: " << measurements.size() << std::endl;

   traccc::default_detector::host detector{m_mr};
   // detector file, material file, grid file
   traccc::io::read_detector(detector,
                             m_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");

   using spacepoint_formation_algorithm
       = traccc::host::spacepoint_formation_algorithm<traccc::default_detector::host>;

   spacepoint_formation_algorithm sf{m_mr};
   spacepoint_formation_algorithm::output_type spacepoints{&m_mr};
   spacepoints = sf(detector, vecmem::get_data(measurements));

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
