> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.

# Running LLaMA models on iOS

On a Mac set-up [ExecuTorch](executorch_setup.md) and open the LLaMA app project with Xcode:

```
open et-build/src/executorch/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
```

Then click the Play button to launch the app in Simulator.

To run on a device, given that you already have it set up for development, you'll need to have a provisioning profile with the [`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement. Just change the app's bundle identifier to whatever matches your provisioning profile with the aforementioned capability enabled.

After the app launched successfully, copy an exported ExecuTorch model (`.pte`) and tokenizer (`.bin`) files to the iLLaMA folder.

For the Simulator, just drap&drop both files onto the Simulator window and save at `On My iPhone > iLLaMA` folder.

For a device, open it in a separate Finder window, navigate to the Files tab, drag&drop both files to the iLLaMA folder and wait till the copying finishes.

Now, follow the app's UI guidelines to pick the model and tokenizer files from the local filesystem and issue a prompt.

Feel free to reuse the [`LLaMARunner`](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA/LLaMARunner/LLaMARunner/Exported) component from the demo app with its C++ [dependencies](https://github.com/pytorch/executorch/tree/main/examples/models/llama2) to your project and give it a spin.

Click the image below to see it in action!

<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>
