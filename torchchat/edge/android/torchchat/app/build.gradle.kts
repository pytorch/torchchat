/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "org.pytorch.torchchat"
  compileSdk = 34

  defaultConfig {
    applicationId = "org.pytorch.torchchat"
    minSdk = 28
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
    externalNativeBuild { cmake { cppFlags += "" } }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
  composeOptions { kotlinCompilerExtensionVersion = "1.4.3" }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}

dependencies {
  implementation("androidx.core:core-ktx:1.9.0")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
  implementation("androidx.activity:activity-compose:1.7.0")
  implementation(platform("androidx.compose:compose-bom:2023.03.00"))
  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.ui:ui-graphics")
  implementation("androidx.compose.ui:ui-tooling-preview")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.appcompat:appcompat:1.6.1")
  implementation("androidx.camera:camera-core:1.3.0-rc02")
  implementation("androidx.constraintlayout:constraintlayout:2.2.0-alpha12")
  implementation("com.facebook.fbjni:fbjni:0.5.1")
  implementation("com.google.code.gson:gson:2.8.6")
  implementation(files("libs/executorch-llama.aar"))
  implementation("com.google.android.material:material:1.12.0")
  implementation("androidx.activity:activity:1.9.0")
  testImplementation("junit:junit:4.13.2")
  androidTestImplementation("androidx.test.ext:junit:1.1.5")
  androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
  androidTestImplementation(platform("androidx.compose:compose-bom:2023.03.00"))
  androidTestImplementation("androidx.compose.ui:ui-test-junit4")
  debugImplementation("androidx.compose.ui:ui-tooling")
  debugImplementation("androidx.compose.ui:ui-test-manifest")
}

tasks.register("setup") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/setup.sh")
      workingDir("../../../../../")
    }
  }
}

tasks.register("setupQnn") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/setup-with-qnn.sh")
      workingDir("../../../../../")
    }
  }
}

tasks.register("download_prebuilt_lib") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/download_prebuilt_lib.sh")
      workingDir("../../../../../")
    }
  }
}
