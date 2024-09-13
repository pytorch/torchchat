> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.

# Executing LLM models on Android

## Option 1: Use ExecuTorch LLAMA Demo App

Check out the [tutorial on how to build an Android app running your
PyTorch models with
ExecuTorch](https://pytorch.org/executorch/main/llm/llama-demo-android.html),
and give your torchchat models a spin.

![Screenshot](https://pytorch.org/executorch/main/_static/img/android_llama_app.png "Android app running Llama model")

Detailed step by step in conjunction with ET Android build, to run on
simulator for Android. `torchchat/utils/scripts/android_example.sh` for running a
model on an Android simulator (on Mac)

## Option 2: Integrate the Java API with your own app

We provide a Java library for you to integrate LLM runner to your own app.
See [this file](https://github.com/pytorch/executorch/blob/main/extension/android/src/main/java/org/pytorch/executorch/LlamaModule.java)
for Java APIs.

To add the Java library to your app, use helper functions `download_aar_library`
from `torchchat/utils/scripts/android_example.sh` to download the prebuilt libraries.

```bash
# my_build_script.sh
source torchchat/utils/scripts/android_example.sh
download_aar_library
```

This will download the AAR to android/torchchat/app/libs/executorch.aar.

In your app working directory (for example executorch/examples/demo-apps/android/LlamaDemo),
copy the AAR to your app libs:
```bash
mkdir -p app/libs
cp ${TORCHCHAT_ROOT}/android/torchchat/app/libs/executorch.aar ${YOUR_APP_ROOT}/app/libs/executorch.aar
```

In your Java app, add the aar file path to your gradle build rule.
```
# app/build.grardle.kts
dependencies {
    implementation(files("libs/executorch.aar"))
}
```

In your Java app, you need to implement [LlamaCallback](https://github.com/pytorch/executorch/blob/main/extension/android/src/main/java/org/pytorch/executorch/LlamaCallback.java).

- `onResult()` is invoked when a token is generated
- `onStats()` is invoked when the `generate()` is done and the tokens/sec is calculated.
- `LlamaModule.generate()` is synchronous. Both are invoked within the same thread as `LlamaModule.generate()`.
  -  You need to run `generate()` in worker thread and handle synchronization within your app.
