# GraphMatching

<div style="display: flex; justify-content: space-around;">
    <img src="resources/readme_pictures/qarma_logo.png" width="150" height="150">
    <img src="resources/readme_pictures/int_logo.png" width="150" height="150">
    <img src="resources/readme_pictures/amu_logo.png" width="150" height="150">
</div>



This toolbox contains scripts to generate Graph from real human cortex graph generated. <br>
Code has been rehabilited and tested by Marius Thorre using Rohit Yadav code. <br>
Demos are available in graph_matching/demos folder.
Test and enjoy :-) <br>
If any questions, please contact me at marius.thorre13@gmail.com


## Installation
First of all, follow these steps to install code
```shell
conda create --name graph_matching python=3.10
```

```shell
conda activate graph_matching
```


```shell
pip install -r requirements.txt
```

## Graph generation
Graphs generated will be store in graph_generated directory.
You can change parameter to generated different graphs.
```shell
python graph_matching/demos/demo_graph_generation.py
```

## Display graph
You can display graphs using this command after generating them.
It will open an HTML file with a graph on sphere mesh.
You can see it in 3d, use your mouth to move around it.
Each node has a color, it's her label.
```shell
python graph_matching/demos/demo_display_graph.py
```

## Compute graph barycenter
From list of graphs generated, you can compute barycenter graph using two different algorithm.
-> Fused Gromov Wasserstein:
```shell
python graph_matching/demos/demo_barycenter_fgw.py
```

-> Unbalanced Fused Gromov Wasserstein:
```shell
python graph_matching/demos/demo_barycenter_fugw.py
```

## Compare pairwise methods 
This script take two graphs in parameter, compute pairwise transport matrix using both barycenter method 
and compare them using euclidian distance.
```shell
python graph_matching/demos/demo_pariwise_graph_analyse.py
```

For more visualisation of noise in graphs. You can run command bellow and select two different html graphs.
## Graph comparaison
```shell
python -m streamlit run graph_matching/demos/demo_graph_comparaison_stApp.py
```



## Authors
- Guillaume AUZIAS (INT)
- François-Xavier DUPE (LIS)
- Marius THORRE (INT, LIS)
- Rohit YADAV (INT, LIS)
- Sylvain TAKERKART (INT)
