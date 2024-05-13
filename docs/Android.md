# Executing LLM models on Android

## Option 1: Use ExecuTorch LLAMA Demo App

Check out the [tutorial on how to build an Android app running your
PyTorch models with
ExecuTorch](https://pytorch.org/executorch/main/llm/llama-demo-android.html),
and give your torchchat models a spin.

![Screenshot](https://pytorch.org/executorch/main/_static/img/android_llama_app.png "Android app running Llama model")

Detailed step by step in conjunction with ET Android build, to run on
simulator for Android. `scripts/android_example.sh` for running a
model on an Android simulator (on Mac)

## Option 2: Integrate the Java API with your own app

We provide a Java library for you to integrate LLM runner to your own app.
See [this file](https://github.com/pytorch/executorch/blob/main/extension/android/src/main/java/org/pytorch/executorch/LlamaModule.java)
for Java APIs.

To add the Java library to your app, use helper functions `download_jar_library`
and `download_jni_library` from `scripts/android_example.sh` to download the
prebuilt libraries.

```bash
# my_build_script.sh
source scripts/android_example.sh
download_jar_library
download_jni_library
```

In your app working directory (for example executorch/examples/demo-apps/android/LlamaDemo),
copy the jar to your app libs:
```bash
mkdir -p app/libs
cp ${TORCHCHAT_ROOT}/build/android/executorch.jar app/libs/executorch.jar
```

In your Java app, add the jar file path to your gradle build rule.
```
# app/build.grardle.kts
dependencies {
    implementation(files("libs/executorch.jar"))
}
```

Then copy the corresponding JNI library to your app:

```bash
mkdir -p app/src/main/jniLibs/arm64-v8a
cp ${TORCHCHAT_ROOT}/build/android/arm64-v8a/libexecutorch_llama_jni.so app/src/main/jniLibs/arm64-v8a
```
