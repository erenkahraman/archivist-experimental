from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="archivist",
    version="0.1.0",
    author="Eren Kahraman",
    author_email="your.email@example.com",
    description="A Python-based image processing and search system with AI-powered analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erenkahraman/archivist-experimental",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.0.1",
        "flask-cors>=3.0.10",
        "Pillow>=9.0.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "torch>=1.10.1",
        "transformers>=4.15.0",
        "python-dotenv>=0.19.2",
        "google-generativeai>=0.3.1",
        "celery>=5.2.3",
        "redis>=4.1.0",
        "werkzeug>=2.0.2",
        "opencv-python>=4.7.0.72",
        "elasticsearch>=8.11.1",
        "elasticsearch-dsl>=8.11.0",
    ],
    entry_points={
        "console_scripts": [
            "archivist=src.app:start_app",
        ],
    },
) 