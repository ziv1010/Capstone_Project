LangSmith tracing is wired up. Set the env vars below before running any stage and traces will be sent automatically.

1) Install deps (if missing): `pip install langsmith langchain-openai langgraph`
2) Export the LangSmith settings (copy/paste from docs):

```
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_PROJECT=<your-project-name>
```

3) Run the pipeline as usual, for example:

```
python run_conversational.py --mode run --task TSK-001
```

Notes:
- The code maps LANGSMITH_* vars to the newer LANGCHAIN_* keys for you, so either prefix works.
- Use `python run_conversational.py --config` to confirm tracing is enabled and which project/endpoint will be used.
- Keep API keys out of source control; set them in your shell or a private .env file.
