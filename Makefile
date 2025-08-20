.PHONY: help install install-backend install-frontend lint format test clean docker-build docker-up docker-down docker-logs docker-clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: install-backend install-frontend ## Install all dependencies

install-backend: ## Install backend dependencies
	cd backend && pip install -r requirements.txt

install-frontend: ## Install frontend dependencies
	cd frontend && pip install -r requirements.txt

lint: ## Run ruff linter
	ruff check .

format: ## Format code with ruff
	ruff format .

format-check: ## Check code formatting
	ruff format --check .

test: ## Run basic tests
	cd backend && python -c "import main; print('Backend imports successfully')"
	cd frontend && python -c "import app; print('Frontend imports successfully')"

clean: ## Clean Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start services with Docker Compose
	docker-compose up -d

docker-down: ## Stop services with Docker Compose
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker images and containers
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

dev-backend: ## Start backend in development mode
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Start frontend in development mode
	cd frontend && streamlit run app.py

health-check: ## Check service health
	@echo "Checking backend health..."
	@curl -s http://localhost:8000/health | jq . || echo "Backend not accessible"
	@echo "Checking frontend health..."
	@curl -s http://localhost:8501/_stcore/health || echo "Frontend not accessible"

all: install lint format-check test ## Run all checks
