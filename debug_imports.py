import sys
import os
from pathlib import Path

# Add the project root to the pythonpath
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

try:
    # Try to import the modules
    print("\nTesting imports...")
    from src.utils.embedding_utils import get_embedding_for_image_id
    print("✅ Successfully imported get_embedding_for_image_id from src.utils.embedding_utils")
    
    from src.utils import get_embedding_for_image_id
    print("✅ Successfully imported get_embedding_for_image_id from src.utils")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    
    # Check if the file exists
    embedding_utils_path = project_root / "src" / "utils" / "embedding_utils.py"
    print(f"\nChecking if file exists at {embedding_utils_path}")
    print(f"File exists: {embedding_utils_path.exists()}")
    
    # List the utils directory
    utils_dir = project_root / "src" / "utils"
    print(f"\nContents of {utils_dir}:")
    for item in utils_dir.iterdir():
        print(f"  - {item.name}")
        
    # Check the __init__.py content
    init_path = utils_dir / "__init__.py"
    if init_path.exists():
        print(f"\nContent of {init_path}:")
        with open(init_path, 'r') as f:
            print(f.read()) 