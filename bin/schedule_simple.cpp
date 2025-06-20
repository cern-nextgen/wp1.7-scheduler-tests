#include <iostream>
#include <string>
#include <cstdlib>
#include <getopt.h> // For command-line argument parsing
#include <iomanip> // Add this include for std::scientific, std::fixed, etc.
#include "Scheduler.hpp"
#include "FirstAlgorithm.hpp"
#include "SecondAlgorithm.hpp"
#include "ThirdAlgorithm.hpp"

// Add pragma to suppress optimization for debugging purposes.
#pragma GCC optimize ("O0")

void printHelp() {
    std::cout << "Usage: schedule_simple [options]\n"
              << "Options:\n"
              << "  --numberOfThreads <N>   Number of threads to use (default: 1, use 0 to use all CPU cores)\n"
              << "  --numberOfStreams <N>   Number of concurrent events (default: 0 = numberOfThreads)\n"
              << "  --warmupEvents <N>      Number of events to process before starting the benchmark (default: 0)\n"
              << "  --maxEvents <N>         Number of events to process (default: -1 for all events in the input file)\n"
              << "  --error-on              Enable error in FirstAlgorithm (default: off)\n"
              << "  --error-event <N>       Set the event ID where the error occurs (default: -1)\n"
              << "  --verbose               Enable verbose output (default: off)\n"
              << "  --help                  Show this help message\n"
              << "  --launchStrategy <single|graph>   Select CUDA launch strategy: 'single' for kernel by kernel launch, 'graph' for CUDA graph launch\n";
}

int main(int argc, char* argv[]) {
    int threads = 1;            // Default number of threads
    int streams = 0;            // Default number of streams (0 = numberOfThreads)
    int warmupEvents = 0;       // Default number of warm up events
    int events = -1;            // Default: all events in input file
    bool errorEnabled = false;  // Default: no error
    int errorEventId = -1;      // Default: no specific event for error
    bool verbose = false;       // Default: no verbose output
    Scheduler::ExecutionStrategy strategy = Scheduler::ExecutionStrategy::CoroutinesSingleLaunch;

    // Define long options
    static struct option long_options[] = {
        {"numberOfThreads", required_argument, nullptr, 1},
        {"numberOfStreams", required_argument, nullptr, 2},
        {"warmupEvents", required_argument, nullptr, 3},
        {"maxEvents", required_argument, nullptr, 4},
        {"error-on", no_argument, nullptr, 5},
        {"error-event", required_argument, nullptr, 6},
        {"verbose", no_argument, nullptr, 7},
        {"help", no_argument, nullptr, 8},
        {"launchStrategy", required_argument, nullptr, 9},
        {nullptr, 0, nullptr, 0}
    };

    // Parse command-line arguments
    int opt;
    int long_index = 0;
    while ((opt = getopt_long(argc, argv, "", long_options, &long_index)) != -1) {
        switch (opt) {
            case 1:
                threads = std::stoi(optarg);
                break;
            case 2:
                streams = std::stoi(optarg);
                break;
            case 3:
                warmupEvents = std::stoi(optarg);
                break;
            case 4:
                events = std::stoi(optarg);
                break;
            case 5:
                errorEnabled = true;
                break;
            case 6:
                errorEventId = std::stoi(optarg);
                break;
            case 7:
                verbose = true;
                break;
            case 8:
                printHelp();
                return 0;
            case 9: {
                std::string value(optarg);
                if (value == "single") {
                    strategy = Scheduler::ExecutionStrategy::CoroutinesSingleLaunch;
                } else if (value == "graph") {
                    strategy = Scheduler::ExecutionStrategy::CoroutinesGraphLaunch;
                } else {
                    std::cerr << "Unknown value for --launchStrategy: " << value << std::endl;
                    return 1;
                }
                break;
            }
            default:
                printHelp();
                return 1;
        }
    }

    // Print configuration
    std::cout << "Starting scheduler with " << threads << " threads, " << streams << " streams, and ";
    if (events == -1) {
        std::cout << "all events";
    } else {
        std::cout << events << " events";
    }
    std::cout << ".\n";
    std::cout << "Warm up events: " << warmupEvents << "\n";
    std::cout << "Error in FirstAlgorithm: " << (errorEnabled ? "enabled" : "disabled")
              << ", event ID: " << errorEventId << "\n";
    std::cout << "Verbose output: " << (verbose ? "enabled" : "disabled") << "\n";

    // Initialize the scheduler
    Scheduler scheduler(threads, streams, strategy);

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

    // Print output in the requested format
    // Assume stats.duration is in milliseconds, convert to seconds
    double time = stats.duration / 1000.0;
    double cpu = 0;// TODO: we would need stats.cpu to match the CMS measurements 
    int maxEvents = stats.events;
    int numberOfThreads = threads;

    std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
              << std::defaultfloat << (time > 0 ? (maxEvents / time) : 0) << " events/s, CPU usage per thread: " << std::fixed
              << std::setprecision(1) << (time > 0 && numberOfThreads > 0 ? (cpu / time / numberOfThreads * 100) : 0.0) << "%" << std::endl;

    return EXIT_SUCCESS;
}
