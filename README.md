# GraphMatching

This toolbox contains scripts to generate Graph from real human cortex graph generated.
Code has been rehabilited and tested by Marius Thorre using Rohit Yadav code.

## Installation
```shell
pip install -e .
```

## Graph generation
Graphs generated will be store in graph_generated directory.
```shell
python graph_matching/generation/script_generation_graphs_with_edges_permutation.py
```

## Display graph
After generated graphs, you can display them using this command:
```shell
python -m notebook graph_matching/display/display_graph.ipynb
```

## Authors
- Guillaume Auzias (INT)
- François-Xavier Dupé (LIS)
- Rohit Yadav (INT, LIS)
- Marius Thorre (INT, LIS)