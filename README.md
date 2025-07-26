# GSMFormer-PPI: Protein-Protein Interaction Prediction using Graph Structure Modeling

A deep learning approach for predicting protein-protein interactions (PPIs) that incorporates protein surface, structural, and graph based features, as well as the implementation of linear projectors and the transformer attention mechanism for the prediction of PPIs.

In order to replicate the results mentioned in the paper, we recommend to follow the next steps:

1. Clone the repository
    ```bash
    git clone https://github.com/ChervovNikita/gsmformer-ppi.git
    cd gsmformer-ppi 
    ```

2. Dataset preparation
    - Our model was trained using the [PINDER dataset](https://github.com/pinder-org/pinder)

    Install the PINDER's packages: 
    ```bash
    pip install pinder[all] 
    ```
    Note: The dataset is large (~700Gb), so please, be carefull with the space on your disk. 

    - Download the BioGRID dataset for cross-validation

    ```bash
    wget -O datasets/BIOGRID-ALL-4.4.238.tab3.zip https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.243/BIOGRID-ALL-4.4.243.tab3.zip
    unzip -d datasets/ datasets/BIOGRID-ALL-4.4.243.tab3.zip
    ``` 

    - Generate the datasets

   For this step, we recommend to follow the step 3 from this repository [SpatialPPIv2](https://github.com/ohuelab/SpatialPPIv2/tree/main?tab=readme-ov-file#3-generate-dataset) 

   This step will generate the train, validation and test sets in CSV format. 

   This was the command that we used: 
   ```bash
   cd SpatialPPIv2
   python scripts/dataset_generator.py --exclude scripts/exclude.txt --split train --biogrid datasets/BIOGRID-4.4.243/BIOGRID-ALL-4.4.243.txt
   ``` 

3. Generate the surface features

    In this work, we used the MaSIF preprocessing framework. The MaSIF algorithm takes 3D protein structures in PDB format as input. It generates a discretised molecular surface, excluding solvent surfaces. This molecular surface is decomposed into overlapping radial patches with a fixed geodesic radius, r = 12 Å. For each point in the mesh, MaSIF calculates and assigns a set of geometric and chemical features, called pre-computed features. MaSIF then embeds these pre-computed features into a numerical vector descriptor. This numerical vector descriptor is then used as input for the GSMFormer model.

    We used the docker container of MaSIF to generate the surface features. You can download the container from [here](https://github.com/LPDI-EPFL/masif/tree/master?tab=readme-ov-file#Docker-container). 
    
    For additional information about the installation of MaSIF, please check the [MaSIF repository](https://github.com/LPDI-EPFL/masif).

    Inside the Docker container, we used the following scripts:

    **Run MaSIF pre computation**
    1. In the directory *data_preparation/00-raw_pdbs*, put your PDB files downloaded from PINDER (in this repository we provided one example). The PDB file 0ant.pdb is a pair of protein chains (left and right chains). 
    2. Run the script *data_prepare.slurm* for pre-computing the surface features using the MaSIF algorithm. 
    We provided the scripts here in the masif_scripts folder. But we advise to check detailed all the scripts in the MaSIF repository.
    This script will create a PDB file for each protein chain in the folder 01-benchmark_pdbs, PLY files for each protein chain in the folder 01-benchmark_surfaces, and precomputed features stored in the folder 04b-precomputation_12A. 

    3. Run the script *compute_descriptors.slurm* for computing the descriptors using the MaSIF algorithm. 
    This script will create the  descriptor files (straight and flipped) for each protein chain in the folder *processed/descriptors*.
    
4. Graph construction 

     In this work, we constructed an amino-acid/residue interaction graph defined as G(V, E), using the 3D coordinates of atoms from PDB files. The nodes or vertices (v ∈ V) are amino acid residues, and the edges (e ∈ E) denote the proximity between residues, including hydrogen bonds,hydrophobic interactions, or spatial proximity associations. The proximity or interaction of two residues was contingent upon the Euclidean distance of any given pair of atoms (one from each residue) being less than the threshold value of 8 angstroms (Å). 

     Please, follow the next steps to generate the graphs: 

     1. Create a folder called *masif_features* at the same level as the *gsmformer-ppi* model.

     2. Inside this folder, you must create two directories called *raw* and *processsed*. In the folder *raw* place all the PDB files of your dataset. 

     3. Run the script `proteins_to_graphs.py`
        ``` bash
        python protein_to_graphs.py
        ``` 
5. Train the model

    1. Put your dataset splits with labels (train, validation, test) in `.npy` format in the same folder *processed/descriptors*.

    2. Create a folder called *logs* inside `gsmformer-ppi`.

    3. Run `bash run_all_simple.sh` to train all the models.
