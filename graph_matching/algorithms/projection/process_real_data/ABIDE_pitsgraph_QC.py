import sys, os
import numpy as np
import xlrd
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import graph_matching.utils.graph_processing as pg
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

if __name__ == "__main__":
    output_dir = os.path.join(project_path, 'data/ABIDE/graph_lh')
    hemi = 'lh'

    # Load subjects' graph data
    graph_files_list = [fil for fil in os.listdir(output_dir) if hemi in fil and fil.endswith('.gpickle')]
    print(f"Number of graph files: {len(graph_files_list)}")

    processed_subjects = list()
    pitgraphs_list = list()
    subjects_list = list()

    for graph_file in graph_files_list:
        print(f"Processing: {graph_file}")
        g = nx.read_gpickle(os.path.join(output_dir, graph_file))
        pitgraphs_list.append(g)
        subjects_list.append(graph_file[:7])

    subjects_mean_depth = [np.mean(pg.graph_nodes_attribute(g, 'depth')) for g in pitgraphs_list]
    subjects_nb_pits = [len(g) for g in pitgraphs_list]

    # Load phenotype data
    subjects_info = os.path.join(project_path, 'data/ABIDE/Phenotypic_V1_0b_traitements_visual_check_GA.xls')
    wb = xlrd.open_workbook(subjects_info)
    sh = wb.sheet_by_name('Feuille1')

    phenotype_colnames = [sh.row(0)[0].value, sh.row(0)[2].value, sh.row(0)[7].value, sh.row(0)[8].value,
                          sh.row(0)[9].value]
    print(phenotype_colnames)

    subjects_list_table = []
    QC_table = []
    diag_table = []
    centers_table = []
    sex_table = []
    age_table = []
    handedness_table = []

    for ind in range(1, sh.nrows):
        subjects_list_table.append('{:07.0f}'.format(sh.cell_value(ind, 1)))
        QC_table.append(float(sh.cell_value(ind, 2)))
        centers_table.append(str(sh.cell_value(ind, 0)))
        age_table.append(float(sh.cell_value(ind, 7)))
        sex_table.append(str(sh.cell_value(ind, 8)))
        handedness_table.append(str(sh.cell_value(ind, 9)))
        diag_table.append('asd' if sh.cell_value(ind, 5) == 1 else 'ctrl')

    QC_table = np.array(QC_table)
    diag_table = np.array(diag_table)
    centers_table = np.array(centers_table)
    sex_table = np.array(sex_table)
    age_table = np.array(age_table)
    handedness_table = np.array(handedness_table)

    print(f"Number of subjects in table: {len(subjects_list_table)}")

    # Keep phenotype only for subjects with data loaded
    inds_keep = [subjects_list_table.index(s) for s in subjects_list if s in subjects_list_table]
    inds_keep = np.array(inds_keep, int)

    QC = QC_table[inds_keep]
    diag = diag_table[inds_keep]
    age = age_table[inds_keep]
    centers = centers_table[inds_keep]
    sex = sex_table[inds_keep]
    handedness = handedness_table[inds_keep]

    subjects_list_check = np.array(subjects_list_table)[inds_keep]
    print(f"Number of matching subjects: {len(set(subjects_list_check).intersection(set(subjects_list)))}")

    # General data plot
    asd_inds = diag == 'asd'
    ctrl_inds = diag == 'ctrl'
    asd_nb_pits = np.array(subjects_nb_pits)[asd_inds]
    asd_mean_depth = np.array(subjects_mean_depth)[asd_inds]
    ctrl_nb_pits = np.array(subjects_nb_pits)[ctrl_inds]
    ctrl_mean_depth = np.array(subjects_mean_depth)[ctrl_inds]

    hist_cent_nb_pits = []
    hist_cent_mean_depth = []
    leg = []

    for ce in set(centers):
        inds_ce = centers == ce
        hist_cent_nb_pits.append(np.array(subjects_nb_pits)[inds_ce])
        hist_cent_mean_depth.append(np.array(subjects_mean_depth)[inds_ce])
        leg.append(f"{np.sum(inds_ce)}  {ce}")
        print(f"{np.sum(inds_ce)}  {ce} mean age =  {np.mean(age[inds_ce])}")

    fig, axes = plt.subplots(3, 2)
    axes[0, 0].hist(subjects_nb_pits)
    axes[0, 0].set_title('Number of pits (all subjects)')
    axes[0, 1].hist(subjects_mean_depth)
    axes[0, 1].set_title('Mean pit depth (all subjects)')
    axes[1, 0].hist([asd_nb_pits, ctrl_nb_pits], label=[f"{np.sum(asd_inds)} ASD", f"{np.sum(ctrl_inds)} CTRL"],
                    density=True)
    axes[1, 0].legend()
    axes[1, 0].set_title('Number of pits by diagnosis')
    axes[1, 1].hist([asd_mean_depth, ctrl_mean_depth], label=[f"{np.sum(asd_inds)} ASD", f"{np.sum(ctrl_inds)} CTRL"],
                    density=True)
    axes[1, 1].legend()
    axes[1, 1].set_title('Mean pit depth by diagnosis')
    axes[2, 0].hist(hist_cent_nb_pits, density=True, label=leg)
    axes[2, 0].legend()
    axes[2, 0].set_title('Number of pits by center')
    axes[2, 1].hist(hist_cent_mean_depth, density=True, label=leg)
    axes[2, 1].set_title('Mean pit depth by center')

    plt.tight_layout()
    plt.show()
