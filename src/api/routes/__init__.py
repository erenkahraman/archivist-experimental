# Import all routes to make them available when routes is imported
from . import image_routes
from . import search_routes
from . import settings_routes

__all__ = ['image_routes', 'search_routes', 'settings_routes'] 