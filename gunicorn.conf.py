# gunicorn.conf.py

bind = "0.0.0.0:8000"
workers = 4  # Adjust this number based on your server's CPU cores
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120  # Timeout for worker processes
