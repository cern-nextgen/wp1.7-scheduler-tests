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
              << "  -t, --threads <N>       Set the number of threads (default: 4)\n"
              << "  -s, --streams <N>       Set the number of CUDA streams (default: 4)\n"
              << "  -e, --events <N>        Set the number of events to process (default: 500)\n"
              << "  -w, --warmup <N>        Set the number of warm up events (default: 0)\n"
              << "  -o, --error-on          Enable error in FirstAlgorithm (default: off)\n"
              << "  -n, --error-event <N>   Set the event ID where the error occurs (default: -1)\n"
              << "  -v, --verbose           Enable verbose output (default: off)\n"
              << "  -h, --help              Show this help message\n";
}

int main(int argc, char* argv[]) {
    int threads = 4;       // Default number of threads
    int streams = 4;       // Default number of streams
    int events = 500;      // Default number of events
    int warmupEvents = 0;  // Default number of warm up events
    bool errorEnabled = false; // Default: no error
    int errorEventId = -1; // Default: no specific event for error
    bool verbose = false;  // Default: no verbose output

    // Define long options
    static struct option long_options[] = {
        {"threads", required_argument, nullptr, 't'},
        {"streams", required_argument, nullptr, 's'},
        {"events", required_argument, nullptr, 'e'},
        {"warmup", required_argument, nullptr, 'w'},
        {"error-on", no_argument, nullptr, 'o'},
        {"error-event", required_argument, nullptr, 'n'},
        {"verbose", no_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    // Parse command-line arguments
    int opt;
    while ((opt = getopt_long(argc, argv, "t:s:e:w:on:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 't':
                threads = std::stoi(optarg);
                break;
            case 's':
                streams = std::stoi(optarg);
                break;
            case 'e':
                events = std::stoi(optarg);
                break;
            case 'w':
                warmupEvents = std::stoi(optarg);
                break;
            case 'o':
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
    std::cout << "Starting scheduler with " << threads << " threads, " << streams << " streams, and " << events << " events.\n";
    std::cout << "Warm up events: " << warmupEvents << "\n";
    std::cout << "Error in FirstAlgorithm: " << (errorEnabled ? "enabled" : "disabled")
              << ", event ID: " << errorEventId << "\n";
    std::cout << "Verbose output: " << (verbose ? "enabled" : "disabled") << "\n";

    // Initialize the scheduler
    Scheduler scheduler(threads, streams);

    // Create the algorithms
    FirstAlgorithm firstAlgorithm(errorEnabled, errorEventId, verbose);
    SecondAlgorithm secondAlgorithm(verbose);
    ThirdAlgorithm thirdAlgorithm(verbose);

    // Add the algorithms to the scheduler
    scheduler.addAlgorithm(firstAlgorithm);
    scheduler.addAlgorithm(secondAlgorithm);
    scheduler.addAlgorithm(thirdAlgorithm);

    // Warm up run if requested
    if (warmupEvents > 0) {
        Scheduler::RunStats warmupStats;
        if (StatusCode status = scheduler.run(warmupEvents, warmupStats); !status) {
            std::cerr << "Warm up run failed: " << status.what() << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Warm up run completed: "
                  << warmupStats.events << " events in "
                  << warmupStats.duration << " ms." << std::endl;
    }

    // Main run
    Scheduler::RunStats stats;
    if (StatusCode status = scheduler.run(events, stats); !status) {
        std::cerr << "Scheduler run failed: " << status.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Scheduler run completed successfully.\n";
    std::cout << "Processed " << stats.events << " events in " << stats.duration << " ms (" << stats.rate << " events/sec)" << std::endl;

    return EXIT_SUCCESS;
}
