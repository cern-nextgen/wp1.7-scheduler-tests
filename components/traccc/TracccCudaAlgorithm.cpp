#include "TracccCudaAlgorithm.hpp"

#include <iostream>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>

#include "Scheduler.hpp"
#include "traccc/clusterization/clustering_config.hpp"


TracccCudaAlgorithm::TracccCudaAlgorithm(int numEvents)
    : m_host_mr{},
      m_cuda_host_mr{},
      m_device_mr{},
      m_mr{m_device_mr, &m_cuda_host_mr},
      m_host_det_descr{m_host_mr},
      m_host_detector{m_host_mr},
      m_device_det_descr{},
      m_device_detector{},
      m_device_detector_view{},
      m_stream{},
      m_copy{m_stream.cudaStream()},
      m_ca_cuda{m_mr, m_copy, m_stream, traccc::clustering_config{256, 16, 8, 256}},
      m_ms_cuda{m_copy, m_stream},
      m_sf_cuda{m_mr, m_copy, m_stream},
      m_sa_cuda{traccc::seedfinder_config{},
                {traccc::seedfinder_config{}},
                traccc::seedfilter_config{},
                m_mr,
                m_copy,
                m_stream},
      m_tp_cuda{m_mr, m_copy, m_stream},
      m_finding_alg_cuda{device_finding_algorithm::config_type{}, m_mr, m_copy, m_stream},
      m_fitting_alg_cuda{device_fitting_algorithm::config_type{}, m_mr, m_copy, m_stream},
      m_cells{},
      m_numEvents{numEvents} {
}


StatusCode TracccCudaAlgorithm::initialize() {
   traccc::io::read_detector_description(m_host_det_descr,
                                         "geometries/odd/odd-detray_geometry_detray.json",
                                         "geometries/odd/odd-digi-geometric-config.json");
   traccc::silicon_detector_description::data host_det_descr_data{
       vecmem::get_data(m_host_det_descr)};
   // Initialize here instead of constructor so that m_device_det_descr gets the correct size.
   m_device_det_descr = traccc::silicon_detector_description::buffer{
       static_cast<traccc::silicon_detector_description::buffer::size_type>(
           m_host_det_descr.size()),
       m_device_mr};
   m_copy(host_det_descr_data, m_device_det_descr)->wait();

   // Detector file, material file, grid file.
   traccc::io::read_detector(m_host_detector,
                             m_host_mr,
                             "geometries/odd/odd-detray_geometry_detray.json",
                             "geometries/odd/odd-detray_material_detray.json",
                             "geometries/odd/odd-detray_surface_grids_detray.json");
   m_device_detector
       = detray::get_buffer(detray::get_data(m_host_detector), m_device_mr, m_copy);
   m_stream.synchronize();
   m_device_detector_view = detray::get_data(m_device_detector);

   m_cells.assign(m_numEvents, traccc::edm::silicon_cell_collection::host{m_host_mr});
   for(std::size_t i{}; auto& cells : m_cells) {
      traccc::io::read_cells(cells, i++, "odd/geant4_10muon_10GeV/", &m_host_det_descr);
   }

   return StatusCode::SUCCESS;
}


AlgorithmBase::AlgCoInterface TracccCudaAlgorithm::execute(EventContext ctx) const {
   const auto& cells = m_cells[ctx.eventNumber % m_numEvents];

   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   traccc::edm::silicon_cell_collection::buffer cells_buffer{
       static_cast<unsigned int>(cells.size()), m_mr.main};
   m_copy(vecmem::get_data(cells), cells_buffer)->ignore();
   auto measurements_cuda_buffer = m_ca_cuda(cells_buffer, m_device_det_descr);
   m_ms_cuda(measurements_cuda_buffer);

   auto spacepoints_cuda_buffer = m_sf_cuda(m_device_detector_view, measurements_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   auto seeds_cuda_buffer = m_sa_cuda(spacepoints_cuda_buffer);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   // Constant B field for the track finding and fitting.
   const traccc::vector3 field_vec = {0.f, 0.f, traccc::seedfinder_config{}.bFieldInZ};
   const detray::bfield::const_field_t field = detray::bfield::create_const_field(field_vec);

   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   auto params_cuda_buffer = m_tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer, field_vec);
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_yield StatusCode::SUCCESS;

   ctx.scheduler->setCudaSlotState(ctx.slotNumber, 0, false);
   auto track_candidates_buffer = m_finding_alg_cuda(
       m_device_detector_view, field, measurements_cuda_buffer, params_cuda_buffer);

   auto track_states_buffer
       = m_fitting_alg_cuda(m_device_detector_view, field, track_candidates_buffer);

   m_stream.synchronize();
   cudaLaunchHostFunc(ctx.stream, notifyScheduler, new EventContext{ctx});
   co_return StatusCode::SUCCESS;
}


StatusCode TracccCudaAlgorithm::finalize() {
   return StatusCode::SUCCESS;
}
