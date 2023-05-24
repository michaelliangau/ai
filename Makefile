format:
	black .

debug:
	uvicorn main:app

deploy:
	sls deploy --stage staging

delete-deploy:
	sls remove --stage staging

build-wheel:
	rm -rf build/ dist/ silicron.egg-info && python setup.py sdist bdist_wheel

upload-wheel:
	twine upload dist/*