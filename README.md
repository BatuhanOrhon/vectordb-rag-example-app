# vectordb-rag-example-app

This project is designed to create sample RAG (Retrieval-Augmented Generation) pipelines using Chroma VectorDB with Python.

## Features

- Chroma VectorDB integration
- Example RAG architectures
- Easily extensible and customizable structure

## Installation

1. Make sure Python 3.8+ is installed.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Chroma VectorDB settings as needed.

## Usage

- Install dependencies (see Installation) then run a basic local Chroma example:

  ```bash
  python src/basic_chroma_example.py
  ```

  A local persistent database will be created under `chroma_db/` (ignored by git).

- You can modify `src/basic_chroma_example.py` to add more documents or experiment with queries.

## Contributing

To contribute, please submit a pull request.

## License

MIT
