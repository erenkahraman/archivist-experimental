"""Celery configuration for asynchronous processing."""
from celery import Celery

celery_app = Celery(
    'archivist',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.task_routes = {
    'tasks.process_image_task': {'queue': 'images'},
    'tasks.analyze_pattern_task': {'queue': 'analysis'}
}

# Optional: Configure task timeouts and retries
celery_app.conf.task_time_limit = 300  # 5 minutes
celery_app.conf.task_soft_time_limit = 240  # 4 minutes
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1 