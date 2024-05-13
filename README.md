# DeGPT
(NDSS 2024) Optimizing Decompiler Output with LLM

## Step 1. Install Dependency

Install the following two package manually.

```
https://github.com/PeiweiHu/cinspector
```

Please also install the following packages by pip.

```
openai==1.28.1
tiktoken==0.2.0
python-levenshtein
```

## Step 2. Setup your API key

Set up your api key in `degpt/chat.py`

```python
api_key = None # configure api_key
api_base = None # configure api_base
assert (api_key and api_base and "Setup your api_key and api_base first")
client = OpenAI(api_key=api_key, base_url=api_base)
```

## Step 3. Do Optimization

```bash
python degpt/role.py -f testcase/fibon out.json
```
