# Silicron
Contextual chat apps.

## Package Deployment

**Usage**
```bash
python example.py
```

The `silicron` folder is built into a package upon running the below commands.

```bash
make build-wheel
make upload-wheel
```
You will be prompted to add your PyPI credentials (michaelliangau)

## Web app

**Debugging**
1. Set up local environment variables:
```bash
export SILICRON_LOCAL_API_ENDPOINT=http://127.0.0.1:8000
```
2. Run the web app
```bash
make debug-setup-local-env
make debug
```

**Deployment**

URLs of our deployed web app:
- [Staging lambda](https://wsesuzvgd0.execute-api.us-east-1.amazonaws.com/staging/)
- Production lambda - TODO

To deploy (change --stage flag to deploy to any named environment)
```bash
make deploy
```

To delete your app
```bash
make delete-deploy
```

This command assumes you have the following installed:
- Docker
- AWS credentials
- [serverless npm package](https://www.npmjs.com/package/serverless) (`npm install -g serverless`)

## Testing
All pytest tests are located in the `tests` folder and are run with the following command:
```bash
make test
```

## Resources
- [Design Doc](https://docs.google.com/document/d/1MfPYqvYliRFHUaQkkjJrplB-LnGcamcLJK97dgilbUY/edit#)
- [FastAPI AWS Lambda Deployment](https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/)

## Note
- pgvector = postgres
- redis vector database = in memory vector db for  caching purposes
