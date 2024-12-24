# Distributed Inference with torchchat

torchchat supports distributed inference for large language models (LLMs) on GPUs seamlessly. 
At present, torchchat supports distributed inference using Python only.

## Installation
The following steps require that you have [Python 3.10](https://www.python.org/downloads/release/python-3100/) installed.

> [!TIP]
> torchchat uses the latest changes from various PyTorch projects so it's highly recommended that you use a venv (by using the commands below) or CONDA.

[skip default]: begin
```bash
git clone https://github.com/pytorch/torchchat.git
cd torchchat
python3 -m venv .venv
source .venv/bin/activate
./install/install_requirements.sh
```
[skip default]: end

[shell default]: ./install/install_requirements.sh

## Login to HF for Downloading Weights
Most models use Hugging Face as the distribution channel, so you will need to create a Hugging Face account. Create a Hugging Face user access token as documented here with the write role.

Log into Hugging Face:

[prefix default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}"

```
huggingface-cli login
```

## Enabling Distributed torchchat Inference

To enable distributed inference, use the option `--distributed`.  In addition, `--tp <num>` and `--pp <num>` 
allow users to specify the types of parallelism to use where tp refers to tensor parallelism and pp to pipeline parallelism.


## Generate Output with Distributed torchchat Inference

To generate output using distributed inference with 4 GPUs, you can use:
```
python3 torchchat.py generate llama3.1 --distributed --tp 2 --pp 2 --prompt "write me a story about a boy and his bear"
```


## Chat with Distributed torchchat Inference

This mode allows you to chat with an LLM in an interactive fashion with distributed Inference.  The following example uses 4 GPUs:

[skip default]: begin
```bash
python3 torchchat.py chat llama3.1 --max-new-tokens 10 --distributed --tp 2 --pp 2
```
[skip default]: end


## A Server with Distributed torchchat Inference

This mode exposes a REST API for interacting with a model.
The server follows the [OpenAI API specification](https://platform.openai.com/docs/api-reference/chat) for chat completions.

To test out the REST API, **you'll need 2 terminals**: one to host the server, and one to send the request.

In one terminal, start the server to run with 4 GPUs:

[skip default]: begin

```bash
python3 torchchat.py server llama3.1 --distributed --tp 2 --pp 2
```
[skip default]: end

<!--
[shell default]: python3 torchchat.py server llama3.1 --distributed --tp 2 --pp 2 & server_pid=$! ; sleep 180 # wait for server to be ready to accept requests
-->

In another terminal, query the server using `curl`. Depending on the model configuration, this query might take a few minutes to respond.

> [!NOTE]
> Since this feature is under active development, not every parameter is consumed. See api/api.py for details on
> which request parameters are implemented. If you encounter any issues, please comment on the [tracking Github issue](https://github.com/pytorch/torchchat/issues/973).

<details>
<summary>Example Query</summary>

Setting `stream` to "true" in the request emits a response in chunks. If `stream` is unset or not "true", then the client will await the full response from the server.

**Example Input + Output**

```
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1",
    "stream": "true",
    "max_tokens": 200,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```
[skip default]: begin
```
{"response":" I'm a software developer with a passion for building innovative and user-friendly applications. I have experience in developing web and mobile applications using various technologies such as Java, Python, and JavaScript. I'm always looking for new challenges and opportunities to learn and grow as a developer.\n\nIn my free time, I enjoy reading books on computer science and programming, as well as experimenting with new technologies and techniques. I'm also interested in machine learning and artificial intelligence, and I'm always looking for ways to apply these concepts to real-world problems.\n\nI'm excited to be a part of the developer community and to have the opportunity to share my knowledge and experience with others. I'm always happy to help with any questions or problems you may have, and I'm looking forward to learning from you as well.\n\nThank you for visiting my profile! I hope you find my information helpful and interesting. If you have any questions or would like to discuss any topics, please feel free to reach out to me. I"}
```

[skip default]: end

<!--
[shell default]: kill ${server_pid}
-->

</details>

[end default]: end
