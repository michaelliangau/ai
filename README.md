# Silicron - Contextual Chat Apps

## Package Development

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

## Web App Development

**Debugging**

1. Set up local environment variables:

```bash
export SILICRON_LOCAL_API_ENDPOINT=http://127.0.0.1:8000
```

This will make the public package route requests to your local endpoint instead of the public one. 
You can also add the above command to your `~/.bashrc` or `~/.zshrc` file which'll run this command everytime you open your shell.


2. Run the web app

```bash
make debug
```

3. To unset local environment variables:
```bash
unset SILICRON_LOCAL_API_ENDPOINT
```

**Deployment**

URLs of our deployed web app:

- [Staging lambda](https://wsesuzvgd0.execute-api.us-east-1.amazonaws.com/staging/)
- Production lambda - TODO


To deploy to staging

```bash
make deploy
```

To delete your staging app

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

Note you need to have the the local webserver running to properly test local package deployments because the package will attempt to make calls to your api endpoints. You can do this by running the following command in a separate terminal window:

```bash
make debug
```

## Gotchas

Sometimes you'll have import package errors when working in subfolders, you can do this to import silicron from above the current directory:

```python3
import sys
sys.path.append('..')
import silicron
```

## Frontend

This is an integration that uses Next.js as the front end and FastAPI as the API backend. Silicron is a use case in which the Next.js application can take advantage of Python AI libraries in the backend.

The Python/FastAPi server is mapped into Next.js app under `/api/`.

This is implemented using `next.config.js` rewrites to map any request to `/api/:path*` to the FastAPI API, which is hosted in the `frontend` folder.

On localhost, the rewrite will be made to the `127.0.0.1:8000` port, which is where the FastAPI server is running.

> **Note** As of right now, two separate development servers have to be running in order this test the API endpoints from the front end.

To get started, cd to the frontend folder from the root directory:

```
cd frontend
```

Install the dependencies:

```
npm install
```

Then run the development server:

```
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.


## Resources

- [Design Doc](https://docs.google.com/document/d/1MfPYqvYliRFHUaQkkjJrplB-LnGcamcLJK97dgilbUY/edit#)
- [FastAPI AWS Lambda Deployment](https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/)
- [Supabase tutorial](https://supabase.com/blog/openai-embeddings-postgres-vector)
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496.pdf)




