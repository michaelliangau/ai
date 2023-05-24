format:
	black .

debug:
	uvicorn main:app

deploy:
	sls deploy --stage staging

delete-deploy:
	sls remove --stage staging