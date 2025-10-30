from setuptools import setup, find_packages

setup(
    name="fraud_detection_common",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="Common utilities and data transformation logic for the fraud detection project.",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
        "pydantic>=1.10.0,<3.0.0",  # Compatible with both Airflow (v1) and modern services (v2)
    ],
)
