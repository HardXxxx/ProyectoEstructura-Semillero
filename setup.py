from setuptools import setup, find_packages

setup(
    name="cold-chain-optimization",
    version="1.0.0",
    description=(
        "Optimización de centros de refrigeración para pesca artesanal "
        "en Puerto Gaitán, Meta, Colombia — mediante teoría de grafos"
    ),
    author="Estructuras de Datos - Universidad",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "networkx>=3.1",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "tabulate>=0.9",
    ],
    extras_require={
        "geo": ["geopandas>=0.13", "shapely>=2.0", "pyproj>=3.5", "fiona>=1.9"],
        "opt": ["pulp>=2.7"],
        "dev": ["pytest>=7.4"],
    },
)

