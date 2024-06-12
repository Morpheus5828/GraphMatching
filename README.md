# GraphMatching

This toolbox contains scripts to generate Graph from real human cortex graph generated. <br>
Code has been rehabilited and tested by Marius Thorre using Rohit Yadav code. <br>
Demos is available in graph_matching/demos folder.
Test and enjoy :-) <br>
If any questions, please contact me at marius.thorre13@gmail.com

## Installation
```shell
pip install -e .
```

## Graph generation
Graphs generated will be store in graph_generated directory.
```shell
python graph_matching/demos/demo_graph_generation.py
```

## Compute graph barycenter
From list of graphs, compute barycenter graph.
```shell
python graph_matching/demos/demos_barycenter.py
```

## Display graph
After generated graphs, you can display them using this command:
```shell
python graph_matching/demos/demo_display_graph.py
```

## Graph comparaison
```shell
python -m streamlit run graph_matching/demos/demo_graph_comparaison.py
```

## Compare pairwise methods
After generated graphs, you can display them using this command:
```shell
python graph_matching/demos/demo_graph_analyse.py
```

## Authors
- Guillaume Auzias (INT)
- François-Xavier Dupé (LIS)
- Marius Thorre (INT, LIS)
- Rohit Yadav (INT, LIS)