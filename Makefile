.PHONY: run run-container gcloud-deploy

run:
	@streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@poetry export --without-hashes > requirements.txt
	@docker build . -t elfragmentador_stlit
	@docker run -p 8080:8080 elfragmentador_stlit

gcloud-deploy:
	@gcloud app deploy app.yaml
