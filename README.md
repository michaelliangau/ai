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

Alternatively, you can also add the above command to your `~/.bashrc` or `~/.zshrc` file which'll run this command everytime you open your shell.

2. Run the web app

```bash
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

## Resources

- [Design Doc](https://docs.google.com/document/d/1MfPYqvYliRFHUaQkkjJrplB-LnGcamcLJK97dgilbUY/edit#)
- [FastAPI AWS Lambda Deployment](https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/)

## Note

- supabase
- pgvector = postgres
- redis vector database = in memory vector db for caching purposes

## Frontend

This is an integration that uses Next.js as the front end and FastAPI as the API backend. Silicron is a use case in which the Next.js application can take advantage of Python AI libraries in the backend.

## Hybrid application struture (\*\*possible roadmap feature)

The Python/FastAPi server is mapped into Next.js app under `/api/`.

This is implemented using `next.config.js` rewrites to map any request to `/api/:path*` to the FastAPI API, which is hosted in the `/silicron_backend` folder.

On localhost, the rewrite will be made to the `127.0.0.1:8000` port, which is where the FastAPI server is running.

> **Note** As of right now, two separate development servers have to be running in order this test the API endpoints from the front end. If you're open to it, I will need to modify the package.json file under "scripts" to concurrently run both servers making it a hybrid application and not two separately hosted applications.

## Running the frontend locally

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

> **Note** On the landing page, click on the sign up button to see the example signup form (inspired/"copied" from [https://assemblyai.com/dashboard/signup](https://assemblyai.com/dashboard/signup))
