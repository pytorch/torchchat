/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.torchchat;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class LlamaModuleTest {
    @Test
    public void LlamaModule() {
        LlamaModule module = new LlamaModule("/data/local/tmp/llama/model.pte", "/data/local/tmp/llama/tokenizer.bin", 0.8f);
        assertEquals(module.load(), 0);
        MyLlamaCallback callback = new MyLlamaCallback();
        // Note: module.generate() is synchronous. Callback happens within the same thread as
        // generate() so when generate() returns, all callbacks are invoked.
        assertEquals(module.generate("Hey", callback), 0);
        assertNotEquals("", callback.result);
    }
}

/**
 * LlamaCallback for testing.
 *
 * Note: onResult() and onStats() are invoked within the same thread as LlamaModule.generate()
 *
 * @see <a href="https://github.com/pytorch/executorch/blob/main/extension/android/src/main/java/org/pytorch/executorch/LlamaCallback.java">LlamaCallback interface guide</a>
 */
class MyLlamaCallback implements LlamaCallback {
    String result = "";
    @Override
    public void onResult(String s) {
        result += s;
    }

    @Override
    public void onStats(float v) {

    }
}
