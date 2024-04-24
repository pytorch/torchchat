# Running LLaMA models on iOS

Check out the [tutorial](https://pytorch.org/executorch/main/llm/llama-demo-ios.html) on how to build the iOS demo app running your
PyTorch models with [ExecuTorch](https://github.com/pytorch/executorch).

Once you can run the app on you device:
1. connect the device to you Mac,
2. copy the model and tokenizer.bin to the iOS Llama app
3. select the tokenizer and model with the `(...)` control (bottom left of screen, to the left of the text entrybox)

Feel free to copy over the [`LLaMARunner`](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA/LLaMARunner/LLaMARunner/Exported) component from the demo app with its C++ [dependencies](https://github.com/pytorch/executorch/tree/main/examples/models/llama2) to your project and give it a spin.

Click the image below to see it in action!

<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>
