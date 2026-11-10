# Instructions

# Environment

The GitHub Copilot SDK requires Python 3.11 or newer. This project uses Python
3.12.3:

```bash
uv venv --python 3.12.3 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Authenticate the Copilot CLI/runtime with a GitHub account that has Copilot
access. Install the [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli), then run:

```bash
copilot auth login
```

The VS Code sign-in is not necessarily shared with the standalone SDK runtime.
See the [Copilot SDK authentication guide](https://github.com/github/copilot-sdk/blob/main/docs/auth/authenticate.md).

## Run

```bash
python main.py
```

The first run downloads the multilingual Sentence Transformers model for local
embeddings. Embeddings and the FAISS index are then cached locally. Answer
generation uses the Copilot SDK; no OpenAI API key is used.

## Use the vector index in Copilot Chat

The workspace skill `/abnt-manual` retrieves excerpts from the local index so
Copilot Chat can answer ABNT questions using the manual. Ask a question about
the manual or invoke `/abnt-manual` directly. The skill uses the existing local
cache, or builds it from the PDF on first use; it does not start the interactive
Copilot SDK app or require SDK authentication.

To search the index manually:

```bash
python .github/skills/abnt-manual/scripts/search.py --query "Como formatar uma citação direta?"
```
