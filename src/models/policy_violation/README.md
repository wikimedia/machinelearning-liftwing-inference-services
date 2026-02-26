# Policy Violation Detection

The policy violation detection inference service uses the gpt-oss-safeguard model to evaluate user content against defined safety policies and classify whether the content violates those policies.

* Model Card: https://arxiv.org/pdf/2508.10925
* Source: https://github.com/roostorg/model-community/tree/142367c67cd8a1d83e293dd0985b50fff0f05b54/gpt
* Model: https://huggingface.co/openai/gpt-oss-safeguard-20b
* Model license: Apache 2.0 License


## How to run locally

In order to run the policy-violation model-server locally, please follow the steps below:

<details>
<summary>1. Docker Compose</summary>

### 1.1. Build
In the first terminal run:
```console
docker compose build policy-violation
```
This will build a policy-violation image with all dependencies installed.

### 1.2. Run
On the the same terminal run the model-server:
```console
docker compose up policy-violation
```

### 1.3. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/policy-violation:predict -X POST \
-d '{ "messages": [{"role": "user", "content": "CLICK HERE TO WIN $10000!!! Visit http://totallylegit.biz NOW before offer expires!!!"}], "developer_prompt": "You are a content safety classifier.\n## Spam Policy\n\n### INSTRUCTIONS\nClassify the following content as spam or not spam. Return JSON: {\"violation\": 0 or 1, \"rationale\": \"brief explanation\"}\n0 = not spam, 1 = spam\n\n### VIOLATES (1)\n- Unsolicited promotional content\n- Repetitive messages\n- Phishing attempts\n- Deceptive links\n\n### SAFE (0)\n- Genuine questions or discussions\n- Relevant information sharing\n- Normal conversation", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95 }' \
-i -H "Content-type: application/json"
```

### 1.4. Remove
If you would like to remove the setup run:
```console
docker compose down -v --rmi all
```

</details>
<details>
<summary>2. Manual setup</summary>

> [!NOTE]
> This model-server is designed to be hosted in a custom-built Docker image that supports vLLM 0.14 and can be found here: https://docker-registry.wikimedia.org/ml/amd-vllm014/tags/
>
> The software stack used in this vLLM image is: ROCm 7.0.0, Torch 2.10.0, MoRi 0.1, FlashAttention 2.8.3, Aiter 0.1.7, and vLLM 0.14. Since we use AMD GPUs on LiftWing, these software packages were built from source to target both MI210 (gfx90a) and MI300X (gfx942) GPUs.
>
> Because these heavy ML dependencies are pre-packaged within the Docker image, the `requirements.txt` file used below only contains the `kserve` dependency needed for the model-server.

### 2.1. Build Python venv and install dependencies
If you are running outside the recommended Docker environment and your system already supports vLLM 0.14, create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/policy_violation/requirements.txt
```

### 2.2. Run the server
We can run the server locally with:
```console
MODEL_NAME=policy-violation MODEL_PATH="openai/gpt-oss-safeguard-20b" TRUST_REMOTE_CODE="True" python3 src/models/policy_violation/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/policy-violation:predict -X POST \
-d '{ "messages": [{"role": "user", "content": "CLICK HERE TO WIN $10000!!! Visit http://totallylegit.biz NOW before offer expires!!!"}], "developer_prompt": "You are a content safety classifier.\n## Spam Policy\n\n### INSTRUCTIONS\nClassify the following content as spam or not spam. Return JSON: {\"violation\": 0 or 1, \"rationale\": \"brief explanation\"}\n0 = not spam, 1 = spam\n\n### VIOLATES (1)\n- Unsolicited promotional content\n- Repetitive messages\n- Phishing attempts\n- Deceptive links\n\n### SAFE (0)\n- Genuine questions or discussions\n- Relevant information sharing\n- Normal conversation", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95 }' \
-i -H "Content-type: application/json"
```
</details>
