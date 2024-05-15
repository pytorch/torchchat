package org.pytorch.torchchat;

import android.content.Context;

import androidx.test.platform.app.InstrumentationRegistry;
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
        LlamaModule module = new LlamaModule("/data/local/tmp/llm/model.pte", "/data/local/tmp/llm/tokenizer.bin", 0.8f);
        assertEquals(module.load(), 0);
        MyLlamaCallback callback = new MyLlamaCallback();
        assertEquals(module.generate("Hey", callback), 0);
        assertNotEquals("", callback.result);
    }
}

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
