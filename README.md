# Silicron
Silicron - Easily extend LLMs with extra context, no code.

[Design Doc](https://docs.google.com/document/d/1MfPYqvYliRFHUaQkkjJrplB-LnGcamcLJK97dgilbUY/edit#)

## Usage

```python3
python main.py
```

## Common utils
`common` contains common utils that are used across projects.
```python3
import sys
sys.path.append("../..")
import common.utils as common_utils
```

## Note
- pgvector = postgres
- redis vector database = in memory vector db for  caching purposes
