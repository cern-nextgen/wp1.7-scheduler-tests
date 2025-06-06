#include <iostream>
#include <string>
#include <cstdlib>
#include <getopt.h> // For command-line argument parsing
#include "Scheduler.hpp"
#include "FirstAlgorithm.hpp"
#include "SecondAlgorithm.hpp"
#include "ThirdAlgorithm.hpp"

// Add pragma to suppress optimization for debugging purposes.
#pragma GCC optimize ("O0")

void printHelp() {
    std::cout << "Usage: schedule_simple [options]\n"
              << "Options:\n"
              << "  --threads <N>       Set the number of threads (default: 4)\n"
              << "  --streams <N>       Set the number of CUDA streams (default: 4)\n"
              << "  --error-on          Enable error in FirstAlgorithm (default: off)\n"
              << "  --error-event <N>   Set the event ID where the error occurs (default: -1)\n"
              << "  --verbose           Enable verbose output (default: off)\n"
              << "  --help              Show this help message\n";
}

int main(int argc, char* argv[]) {
    int threads = 4; // Default number of threads
    int streams = 4; // Default number of streams
    bool errorEnabled = false; // Default: no error
    int errorEventId = -1; // Default: no specific event for error
    bool verbose = false; // Default: no verbose output

    // Define long options
    static struct option long_options[] = {
        {"threads", required_argument, nullptr, 't'},
        {"streams", required_argument, nullptr, 's'},
        {"error-on", no_argument, nullptr, 'e'},
        {"error-event", required_argument, nullptr, 'n'},
        {"verbose", no_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    // Parse command-line arguments
    int opt;
    while ((opt = getopt_long(argc, argv, "t:s:en:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 't':
                threads = std::stoi(optarg);
                break;
            case 's':
                streams = std::stoi(optarg);
                break;
            case 'e':
                errorEnabled = true;
                break;
            case 'n':
                errorEventId = std::stoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                printHelp();
                return 0;
            default:
                printHelp();
                return 1;
        }
    }

    // Print configuration
    std::cout << "Starting scheduler with " << threads << " threads and " << streams << " streams.\n";
    std::cout << "Error in FirstAlgorithm: " << (errorEnabled ? "enabled" : "disabled")
              << ", event ID: " << errorEventId << "\n";
    std::cout << "Verbose output: " << (verbose ? "enabled" : "disabled") << "\n";

    // Initialize the scheduler
    Scheduler scheduler(/*events=*/500, threads, streams);

    // Create the algorithms
    FirstAlgorithm firstAlgorithm(errorEnabled, errorEventId, verbose);
    SecondAlgorithm secondAlgorithm(verbose);
    ThirdAlgorithm thirdAlgorithm(verbose);

    // Add the algorithms to the scheduler
    scheduler.addAlgorithm(firstAlgorithm);
    scheduler.addAlgorithm(secondAlgorithm);
    scheduler.addAlgorithm(thirdAlgorithm);

    // Run the scheduler
    auto w = scheduler.run().what();
    std::cout << "Final scheduler status: " << w << std::endl;

    return EXIT_SUCCESS;
}
