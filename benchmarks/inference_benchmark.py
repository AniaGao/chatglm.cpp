import time
import psutil
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.chatglm import ChatGLM


def benchmark_inference(model_path, num_iterations):
    model = ChatGLM(model_path)
    prompt = "你好"
    
    total_time = 0
    total_memory = 0
    
    process = psutil.Process(os.getpid())
    
    for i in range(num_iterations):
        start_time = time.time()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss
        
        model.stream_generate(prompt, max_length=128) # Using Stream generate to get more accurate tokens/sec

        end_time = time.time()

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory  # Memory usage in bytes
        
        iteration_time = end_time - start_time
        
        total_time += iteration_time
        total_memory += memory_used

        # Clear CUDA cache to avoid OOM errors during benchmarking
        # If CUDA isn't available, this will do nothing
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    average_time = total_time / num_iterations
    average_memory = total_memory / num_iterations
    
    # Estimate 'tokens/second'. Since generating '你好' and 
    # the response. Make a rough guess of 30.
    average_tokens_per_second = 30/average_time

    print(f"Average time per iteration: {average_time:.4f} seconds")
    print(f"Average memory usage per iteration: {average_memory / (1024 * 1024):.2f} MB")
    print(f"Estimated tokens per second: {average_tokens_per_second:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ChatGLM inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ChatGLM model.")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of inference iterations.")
    
    args = parser.parse_args()
    
    benchmark_inference(args.model_path, args.num_iterations)
