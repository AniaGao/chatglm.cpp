import chatglm

model = chatglm.ChatGLM()
model.load_model("model.weights") # Replace with the actual path to your weights file

input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
output_data = model.forward(input_data)

print(f"Input: {input_data}")
print(f"Output: {output_data}")