# Multimodal Models

Released on September 25th, 2024, **Llama3.2 11B Vision** is torchchat's first multimodal model.

This page goes over the different commands you can run with LLama 3.2 11B Vision.

## Model Access

> [!NOTE]
> While the commands refer to the model as some variant of "Llama 3.2 11B Vision",
> the underlying checkpoint used is based off the "Instruct" variant of the model.

**Llama3.2 11B Vision** is available via both [Hugging Face](https://huggingface.co/meta-llama) and [directly from Meta](https://www.llama.com/).

While we strongly encourage you to use the Hugging Face checkpoint (which is the default for torchchat when utilizing the commands with the argument `llama3.2-11B`), we also provide support for manually providing the checkpoint. This can be done by replacing the `llama3.2-11B` argument in the commands below with the following:

```
--checkpoint-path <file.pth> --tokenizer-path <tokenizer.model> --params-path torchchat/model_params/Llama-3.2-11B-Vision.json
```

##  Generation

**We are currently debugging Multimodal Inference on MPS and will have updates soon. In the meantime, when testing on Mac, please set `--device cpu`**

This generates text output based on a text prompt and (optional) image prompt.

```
python torchchat.py generate llama3.2-11B --prompt "What's in this image?" --image-prompt assets/dog.jpg
```

## Server
This mode exposes a REST API for interacting with a model.
The server follows the [OpenAI API specification](https://platform.openai.com/docs/api-reference/chat) for chat completions.

To test out the REST API, **you'll need 2 terminals**: one to host the server, and one to send the request.
In one terminal, start the server

[skip default]: begin

```bash
python3 torchchat.py server llama3.2-11B
```
[skip default]: end

In another terminal, query the server using `curl`. This query might take a few minutes to respond.

**We are currently debugging the server integration and will have updated examples shortly.**

## Browser

This command opens a basic browser interface for local chat by querying a local server.

First, follow the steps in the Server section above to start a local server. Then, in another terminal, launch the interface. Running the following will open a tab in your browser.

[skip default]: begin

```
streamlit run torchchat/usages/browser.py
```

**We are currently debugging the browser integration and will have updated examples shortly.**

---

# Future Work

One of the goals of torchchat is to support various execution modes for every model. The following are execution modes that will be supported for **Llama3.2 11B Vision** in the near future:

- **[torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)**: Optimize inference via JIT Compilation
- **[AOTI](https://pytorch.org/blog/pytorch2-2/)**: Enable pre-compiled and C++ inference
- **[ExecuTorch](https://github.com/pytorch/executorch)**: On-device (Edge) inference

In addition, we are in the process of integrating with [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) for multimodal model evaluation.
