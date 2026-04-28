.PHONY: run-streamlit run-api run-frontend

run-streamlit:
	streamlit run app.py --server.port 8501

run-api:
	uvicorn api.main:app --reload --port 8000

run-frontend:
	cd frontend && npm run dev
