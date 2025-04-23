import sys
sys.path.append('../python')
import chatglm

# Example usage
if __name__ == '__main__':
    # Initialize the ChatGLM model
    model_path = "/path/to/your/model"
    model = chatglm.ChatGLM(model_path)

    #Test the py function is working
    print("C++ call: " + chatglm.test_func())

    # Generate text based on a prompt
    prompt = "Write a short story about a cat named mittens"
    generated_text = model.generate(prompt, max_length=100)

    # Print the generated text
    print(f"Generated text: {generated_text}")
