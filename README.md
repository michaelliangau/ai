# Silicron
Silicron - Easily extend LLMs with extra context, no code.

## Usage

Package internals
```bash
python example.py
```

Web app debugging
```bash
cd app
uvicorn main:app
```

## Deployment

1. Make changes
2. Run the following bash comands

To deploy (change --stage flag to deploy to any named environment)
```bash
sls deploy --stage staging 
```

To delete your app
```bash
sls remove --stage staging
```

This command assumes you have the following installed:
- Docker
- AWS credentials
- [serverless npm package](https://www.npmjs.com/package/serverless) (`npm install -g serverless`)

## Resources
- [Design Doc](https://docs.google.com/document/d/1MfPYqvYliRFHUaQkkjJrplB-LnGcamcLJK97dgilbUY/edit#)
- [FastAPI AWS Lambda Deployment](https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/)

## Note
- pgvector = postgres
- redis vector database = in memory vector db for  caching purposes
