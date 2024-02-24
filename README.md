# DeGPT
(NDSS 2024) Optimizing Decompiler Output with LLM

## Step 1. Install Dependency

Install the following two package manually.

```
https://github.com/PeiweiHu/cinspector
https://github.com/PeiweiHu/chatmanager
```

Please also install the following packages by pip.

```
openai==0.27.6
tiktoken==0.2.0
```

## Step 2. Setup your API key

Set up your api key in `degpt/role.py`

```python
# ------------------- config here ---------------------

PROMPT_PATH = os.path.join(DIR, 'prompt.json')
CHATGPT_API_KEY = {Input your API key here}
ChatSetup.temperature = 0.2
assert (CHATGPT_API_KEY and "Pleae set api key")
# -----------------------------------------------------
```


## Step 3. Do Optimization

```bash
python degpt/role.py -f testcase/fibon out.json
```

This is a bare version with the core component. More data such as metrics is preparing....
