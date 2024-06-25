from setuptools import setup, find_packages

setup(
    name='graph_matching',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'matplotlib>=3.8.4',
        'networkx>=3.3',
        'pandas>=2.2.2',
        'pillow>=10.3.0',
        'scipy>=1.13.0',
        'seaborn>=0.13.2',
        'tqdm>=4.66.4',
        'trimesh>=4.3.2',
        'notebook>=7.2.0',
        'jupyter>=1.0.0',
        'scikit-learn>=1.0',
        'streamlit>=1.35.0',
        'slam>=0.6.1',
        'gdist>=2.1.0',
        'nibabel>=3.2.2',
        'pytest>=8.2.2',
    ],
)