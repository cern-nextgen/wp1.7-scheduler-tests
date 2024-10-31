
// Local include(s).
#include "TracccAlgorithm.hpp"

StatusCode TracccAlgorithm::initialize()
{
   return StatusCode::SUCCESS;
}

AlgorithmBase::AlgCoInterface TracccAlgorithm::execute(EventContext /*ctx*/) const
{
   co_return StatusCode::SUCCESS;
}

StatusCode TracccAlgorithm::finalize()
{
   return StatusCode::SUCCESS;
}

const std::vector<std::string> &TracccAlgorithm::dependencies() const
{
   static std::vector<std::string> deps = {};
   return deps;
}

const std::vector<std::string> &TracccAlgorithm::products() const
{
   static std::vector<std::string> prods = {};
   return prods;
}
