# benchmarks/benchmark.py
import time

def run_benchmark():
    start_time = time.time()
    # Simulate some workload
    for _ in range(1000000):
        pass  # Replace with actual benchmarking code
    end_time = time.time()
    print(f"Benchmark took: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
