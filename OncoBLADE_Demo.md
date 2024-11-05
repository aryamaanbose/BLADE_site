# OncoBLADE DEMO


In this notebook, a demo of OncoBLADE is provided.

In this demo we will use a subset of the Non Small Cell Lung Carcinoma (NSCLC) samples from the manuscript to show:
1) Cell fraction estimation
2) Cell-type specific gene expression estimation
3) Downstream analysis

### Move to the right directory and load necessary modules


```python
import sys, os
os.chdir('..') ## Set directory to one up
from Deconvolution.OncoBLADE import Framework_Iterative
from Deconvolution.OncoBLADE import Purify_AllGenes
import numpy as np
from numpy import transpose as t
import itertools
import pickle
from scipy.optimize import nnls
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from scipy.stats import zscore

# modules for visualization
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

```

## Introduction: Deconvolution with OncoBLADE

OncoBLADE is a Bayesian RNA deconvolution method which is specifically tailored to accurately estimate both cell fractions in parallel with cell type-specific gene expression profiles from bulk RNA of solid tumors. Like any deconvolution method, OncoBLADE assumes that the observed bulk RNA profile ($y_{ij}$) of gene $j$ and sample $i$ can be deconvoluted into two hidden variables:

$$y_{ij} = \sum_{t=1}^{T }f_{i}^{t}x_{ij}^{t}$$

Where, $f_{i}^{t}$ is the fraction of cell type $t$ in sample $i$ and $x_{ij}^{t}$ is the expression of gene $j$ in sample $i$ and cell type $t$.

Besides integrating prior knowledge from single cell RNAseq signatures ($x_{ij}^{t}$) like other methods, OncoBLADE also efficiently integrates prior knowledge on fractions ($f_{i}^{t}$) and in particular the malignant cell fraction ($\bar{f}_{i}^{x}$).

## Creation of a single cell RNAseq signature
<div>
<img src="Demo_Figures/Prior_Signature.png" width="500"/>
</div>

OncoBLADE uses a single cell RNAseq (scRNAseq) signature $x_{ij}^{t}$ as prior knowledge for bulk RNA deconvolution.
Ideally the scRNAseq dataset is from the same type of tumor tissue as the bulk RNAseq, as is the case here with NSCLC.

The signature in OncoBLADE consists of two main parameters, namely the expected cell type sepcific-gene expression level $\mu_{j}^{t}$,  and the expected gene expression variability $\lambda_{j}^{t}$.

To create the signature we use the phenotyped scRNAseq to calculate a mean (Mu) and expected gene expression variability (Omega) per gene per celltype. The expected gene expression variability was estimated by fitting a LOWESS curve to the mean-variance trend measured in the scRNAseq data using the fitTrendVar function implemented in scran.

For this demo we use the same signature as was used in the NSCLC in silico experiments of the manuscript.


```python
# Load NSCLC signature
with open('data/NewSignature_matrices_final_celltype_k1000.pickle', "rb") as infile:
    signature = pickle.load(infile)

## Extract Mu & Omega (nGenes x nCelltypes)
Mu = pd.DataFrame(signature['Mu'], index = signature['Genelist'], columns = signature['celltype_list'])
Omega = pd.DataFrame(signature['Omega'], index = signature['Genelist'], columns = signature['celltype_list'])

## Add pseudocount to Omega as 0 will otherwise give computational problems
Omega = Omega + 0.01
## Print first 5 rows
Mu.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cancer_cell</th>
      <th>Lung_endothelial_cell</th>
      <th>Fibroblast</th>
      <th>Macrophage</th>
      <th>Plasma_cell</th>
      <th>Lung_epithelial_cell</th>
      <th>CD4_proliferating_T_cell</th>
      <th>Monocyte</th>
      <th>Mast_cell</th>
      <th>DC</th>
      <th>CD8_effector_T_cell</th>
      <th>B_cell</th>
      <th>Neutrophil</th>
      <th>Alveolar_cell</th>
      <th>CD4_Th17_like_cell</th>
      <th>CD4_Treg</th>
      <th>NK_cell</th>
      <th>CD4_naive_T_cell</th>
      <th>CD8_exhausted_T_cell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A1BG</th>
      <td>0.022682</td>
      <td>0.072445</td>
      <td>0.147478</td>
      <td>0.065907</td>
      <td>0.112361</td>
      <td>0.056572</td>
      <td>0.066149</td>
      <td>0.116034</td>
      <td>0.191235</td>
      <td>0.154975</td>
      <td>0.118646</td>
      <td>0.142474</td>
      <td>0.000000</td>
      <td>0.019581</td>
      <td>0.153735</td>
      <td>0.157920</td>
      <td>0.062916</td>
      <td>0.134535</td>
      <td>0.174374</td>
    </tr>
    <tr>
      <th>A2M</th>
      <td>0.043105</td>
      <td>2.459644</td>
      <td>1.926918</td>
      <td>0.939422</td>
      <td>0.051905</td>
      <td>0.045449</td>
      <td>0.046703</td>
      <td>0.150999</td>
      <td>0.036530</td>
      <td>0.338606</td>
      <td>0.046928</td>
      <td>0.040528</td>
      <td>0.026962</td>
      <td>0.029398</td>
      <td>0.051364</td>
      <td>0.035965</td>
      <td>0.045805</td>
      <td>0.023281</td>
      <td>0.045238</td>
    </tr>
    <tr>
      <th>A4GALT</th>
      <td>0.201694</td>
      <td>0.187970</td>
      <td>0.162045</td>
      <td>0.005898</td>
      <td>0.006781</td>
      <td>0.104433</td>
      <td>0.007211</td>
      <td>0.004006</td>
      <td>0.004332</td>
      <td>0.002551</td>
      <td>0.001266</td>
      <td>0.020844</td>
      <td>0.001368</td>
      <td>0.005708</td>
      <td>0.003677</td>
      <td>0.006159</td>
      <td>0.000000</td>
      <td>0.000491</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>AAAS</th>
      <td>0.098043</td>
      <td>0.058089</td>
      <td>0.082713</td>
      <td>0.055850</td>
      <td>0.052889</td>
      <td>0.086882</td>
      <td>0.102814</td>
      <td>0.027392</td>
      <td>0.038696</td>
      <td>0.065006</td>
      <td>0.034847</td>
      <td>0.034258</td>
      <td>0.004479</td>
      <td>0.075326</td>
      <td>0.036957</td>
      <td>0.053784</td>
      <td>0.041033</td>
      <td>0.026553</td>
      <td>0.050120</td>
    </tr>
    <tr>
      <th>AACS</th>
      <td>0.097181</td>
      <td>0.028963</td>
      <td>0.043312</td>
      <td>0.025011</td>
      <td>0.026961</td>
      <td>0.053326</td>
      <td>0.032932</td>
      <td>0.013841</td>
      <td>0.023488</td>
      <td>0.027993</td>
      <td>0.018962</td>
      <td>0.014733</td>
      <td>0.002787</td>
      <td>0.054339</td>
      <td>0.016715</td>
      <td>0.023454</td>
      <td>0.031417</td>
      <td>0.023642</td>
      <td>0.021126</td>
    </tr>
  </tbody>
</table>
</div>




```python
Omega.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cancer_cell</th>
      <th>Lung_endothelial_cell</th>
      <th>Fibroblast</th>
      <th>Macrophage</th>
      <th>Plasma_cell</th>
      <th>Lung_epithelial_cell</th>
      <th>CD4_proliferating_T_cell</th>
      <th>Monocyte</th>
      <th>Mast_cell</th>
      <th>DC</th>
      <th>CD8_effector_T_cell</th>
      <th>B_cell</th>
      <th>Neutrophil</th>
      <th>Alveolar_cell</th>
      <th>CD4_Th17_like_cell</th>
      <th>CD4_Treg</th>
      <th>NK_cell</th>
      <th>CD4_naive_T_cell</th>
      <th>CD8_exhausted_T_cell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A1BG</th>
      <td>0.036189</td>
      <td>0.098136</td>
      <td>0.181896</td>
      <td>0.080758</td>
      <td>0.155971</td>
      <td>0.071124</td>
      <td>0.088225</td>
      <td>0.161703</td>
      <td>0.270806</td>
      <td>0.162115</td>
      <td>0.208026</td>
      <td>0.197206</td>
      <td>0.010000</td>
      <td>0.035627</td>
      <td>0.230141</td>
      <td>0.223709</td>
      <td>0.098819</td>
      <td>0.184399</td>
      <td>0.219604</td>
    </tr>
    <tr>
      <th>A2M</th>
      <td>0.059641</td>
      <td>1.187655</td>
      <td>1.226067</td>
      <td>0.803350</td>
      <td>0.078141</td>
      <td>0.059110</td>
      <td>0.065231</td>
      <td>0.205610</td>
      <td>0.061031</td>
      <td>0.324161</td>
      <td>0.088831</td>
      <td>0.064017</td>
      <td>0.076573</td>
      <td>0.048453</td>
      <td>0.084862</td>
      <td>0.059597</td>
      <td>0.074665</td>
      <td>0.040504</td>
      <td>0.065863</td>
    </tr>
    <tr>
      <th>A4GALT</th>
      <td>0.236096</td>
      <td>0.233397</td>
      <td>0.198340</td>
      <td>0.016346</td>
      <td>0.018961</td>
      <td>0.122678</td>
      <td>0.018528</td>
      <td>0.015260</td>
      <td>0.016052</td>
      <td>0.012545</td>
      <td>0.012126</td>
      <td>0.037782</td>
      <td>0.013378</td>
      <td>0.017474</td>
      <td>0.015360</td>
      <td>0.018494</td>
      <td>0.010000</td>
      <td>0.010643</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>AAAS</th>
      <td>0.121755</td>
      <td>0.080672</td>
      <td>0.107298</td>
      <td>0.069993</td>
      <td>0.079423</td>
      <td>0.103837</td>
      <td>0.131474</td>
      <td>0.045965</td>
      <td>0.064057</td>
      <td>0.074844</td>
      <td>0.068537</td>
      <td>0.055660</td>
      <td>0.021060</td>
      <td>0.108113</td>
      <td>0.063866</td>
      <td>0.084168</td>
      <td>0.067927</td>
      <td>0.044791</td>
      <td>0.071892</td>
    </tr>
    <tr>
      <th>AACS</th>
      <td>0.120793</td>
      <td>0.045239</td>
      <td>0.060974</td>
      <td>0.036898</td>
      <td>0.045523</td>
      <td>0.067618</td>
      <td>0.048946</td>
      <td>0.028173</td>
      <td>0.042812</td>
      <td>0.037925</td>
      <td>0.041852</td>
      <td>0.029637</td>
      <td>0.016882</td>
      <td>0.080934</td>
      <td>0.034362</td>
      <td>0.042344</td>
      <td>0.054352</td>
      <td>0.040977</td>
      <td>0.036088</td>
    </tr>
  </tbody>
</table>
</div>



## Preparation of Bulk RNAseq and expected tumor fraction
<div>
<img src="Demo_Figures/Prior_Fraction.png" width="500"/>
</div>

The main innovation of OncoBLADE with respect to other RNA deconvolution methods, is that it allows the user to inform it with cell fraction estimates. OncoBLADE can for example be informed by DNA-derived malignant fraction estimates $\bar{f}_{i}^{x}$, making the rest of the deconvolution task significantly easier.
Besides malignant cell fraction estimates, you can also include information on other cell types if you have it. For example if you have Immunohistochemistry stainings on your samples.

Below we show how to prepare your prior fraction expectation and the bulk RNAseq before performing the deconvolution.



```python
## Load in bulk RNAseq, first 5 patients are LUAD, last 5 are LUSC
bulk = pd.read_csv('data/Transcriptome_matrix_subset.txt', sep = '\t')

## Normalize bulk RNAseq to the same scale as the RNAseq, we use counts per 10k.
## Do not Log normalize, this is done within OncoBLADE
samples = [s for s in bulk.columns if 'TCGA' in s]
bulk[samples] = bulk[samples].apply(lambda x: x / sum(x) * 10000)

# set index to gene symbols and put normalized bulk in Y
Y =  bulk.set_index('symbol')
Y.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TCGA-91-6840</th>
      <th>TCGA-78-8655</th>
      <th>TCGA-NJ-A7XG</th>
      <th>TCGA-38-4625</th>
      <th>TCGA-38-4627</th>
      <th>TCGA-85-A4QQ</th>
      <th>TCGA-56-8628</th>
      <th>TCGA-63-A5M9</th>
      <th>TCGA-77-8128</th>
      <th>TCGA-34-5231</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TSPAN6</th>
      <td>1.643189</td>
      <td>1.005787</td>
      <td>0.986710</td>
      <td>1.015124</td>
      <td>1.559100</td>
      <td>0.308683</td>
      <td>0.829310</td>
      <td>0.904077</td>
      <td>0.727355</td>
      <td>0.653832</td>
    </tr>
    <tr>
      <th>TNMD</th>
      <td>0.001019</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000437</td>
      <td>0.000000</td>
      <td>0.000496</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000099</td>
    </tr>
    <tr>
      <th>DPM1</th>
      <td>0.478754</td>
      <td>0.269087</td>
      <td>0.275330</td>
      <td>0.673312</td>
      <td>0.293269</td>
      <td>0.452470</td>
      <td>0.283463</td>
      <td>0.516275</td>
      <td>0.456260</td>
      <td>0.378602</td>
    </tr>
    <tr>
      <th>SCYL3</th>
      <td>0.185521</td>
      <td>0.295398</td>
      <td>0.194395</td>
      <td>0.117442</td>
      <td>0.105425</td>
      <td>0.106066</td>
      <td>0.144088</td>
      <td>0.172375</td>
      <td>0.179041</td>
      <td>0.200384</td>
    </tr>
    <tr>
      <th>C1orf112</th>
      <td>0.148485</td>
      <td>0.084015</td>
      <td>0.056444</td>
      <td>0.227574</td>
      <td>0.054897</td>
      <td>0.111430</td>
      <td>0.073408</td>
      <td>0.320247</td>
      <td>0.160673</td>
      <td>0.174342</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Prepare prior expectation
# Load ACE tumor fraction estimates
purities = pd.read_csv('data/ACE_Tumor_purities_squaremodel.tsv', sep = "\t")

# obtain list of expected tumor purities for the 10 samples
expected_tumor_purity =pd.merge(pd.DataFrame({'sample': samples }),purities, how = 'left').ACE.tolist()

# Intialize Expectation (Nsample x Ncell with None for non-tumor celltypes)
Expectation = np.zeros((len(samples), len(signature['celltype_list']))) + np.nan

# iterate over samples
for i in range(len(samples)):
    # iterate over celltypes
    for j,celltype in enumerate(signature['celltype_list']):
        if celltype in ['Cancer_cell', 'Tumor cell']:
            # fetch true tumor purity and add to array
            Expectation[i,j] = expected_tumor_purity[i]
        else:
            pass

pd.DataFrame(Expectation, index = bulk.set_index('symbol').columns, columns = signature['celltype_list'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cancer_cell</th>
      <th>Lung_endothelial_cell</th>
      <th>Fibroblast</th>
      <th>Macrophage</th>
      <th>Plasma_cell</th>
      <th>Lung_epithelial_cell</th>
      <th>CD4_proliferating_T_cell</th>
      <th>Monocyte</th>
      <th>Mast_cell</th>
      <th>DC</th>
      <th>CD8_effector_T_cell</th>
      <th>B_cell</th>
      <th>Neutrophil</th>
      <th>Alveolar_cell</th>
      <th>CD4_Th17_like_cell</th>
      <th>CD4_Treg</th>
      <th>NK_cell</th>
      <th>CD4_naive_T_cell</th>
      <th>CD8_exhausted_T_cell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-91-6840</th>
      <td>0.55</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-78-8655</th>
      <td>0.53</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-NJ-A7XG</th>
      <td>0.43</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-38-4625</th>
      <td>0.20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-38-4627</th>
      <td>0.20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-85-A4QQ</th>
      <td>0.28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-56-8628</th>
      <td>0.41</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-63-A5M9</th>
      <td>0.64</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-77-8128</th>
      <td>0.43</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TCGA-34-5231</th>
      <td>0.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Subset signature & bulk on selected genes for deconvolution

OncoBLADE like other RNA deconvolution tools uses a selection of genes for its deconvolution due to the otherwise enormous complexity.

For this we want genes discrimating the cell types, $J_{marker}$, which are to be deconvoluted. We use AutoGeneS for this and use the genes it selected on our scRNAseq here.


```python
## Load autogenes which was saved in the signature
AutoGeneS = signature['AutoGeneS']['Unnamed: 0'].tolist()

## Find common AutoGeneS with the bulk data
common_AutoGeneS = [value for value in AutoGeneS if value in Y.index.to_list()]

## Subset Mu, Omega & bulk on AutoGeneS
Mu_AutoGeneS = Mu.loc[common_AutoGeneS,]
Omega_AutoGeneS = Omega.loc[common_AutoGeneS,]
Y_AutoGeneS = Y.loc[common_AutoGeneS,]
Y_AutoGeneS
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TCGA-91-6840</th>
      <th>TCGA-78-8655</th>
      <th>TCGA-NJ-A7XG</th>
      <th>TCGA-38-4625</th>
      <th>TCGA-38-4627</th>
      <th>TCGA-85-A4QQ</th>
      <th>TCGA-56-8628</th>
      <th>TCGA-63-A5M9</th>
      <th>TCGA-77-8128</th>
      <th>TCGA-34-5231</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A2M</th>
      <td>10.682426</td>
      <td>10.446552</td>
      <td>1.925969</td>
      <td>5.821721</td>
      <td>23.110129</td>
      <td>0.628959</td>
      <td>13.675426</td>
      <td>3.468265</td>
      <td>2.097612</td>
      <td>3.076826</td>
    </tr>
    <tr>
      <th>ABCA1</th>
      <td>0.746502</td>
      <td>0.447581</td>
      <td>0.163208</td>
      <td>0.999605</td>
      <td>0.387482</td>
      <td>0.533274</td>
      <td>0.677782</td>
      <td>0.248778</td>
      <td>0.751002</td>
      <td>0.190345</td>
    </tr>
    <tr>
      <th>ABCA3</th>
      <td>0.451231</td>
      <td>1.060800</td>
      <td>3.984150</td>
      <td>1.075697</td>
      <td>0.364474</td>
      <td>0.055542</td>
      <td>4.889802</td>
      <td>0.343559</td>
      <td>0.237948</td>
      <td>0.234378</td>
    </tr>
    <tr>
      <th>AC022182.3</th>
      <td>0.001019</td>
      <td>0.000598</td>
      <td>0.000383</td>
      <td>0.000801</td>
      <td>0.001747</td>
      <td>0.000173</td>
      <td>0.002480</td>
      <td>0.001531</td>
      <td>0.000000</td>
      <td>0.000795</td>
    </tr>
    <tr>
      <th>AC133644.2</th>
      <td>0.008155</td>
      <td>0.011660</td>
      <td>0.001913</td>
      <td>0.007209</td>
      <td>0.016455</td>
      <td>0.002768</td>
      <td>0.003968</td>
      <td>0.008168</td>
      <td>0.008023</td>
      <td>0.004373</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZMIZ1</th>
      <td>2.380857</td>
      <td>0.627272</td>
      <td>0.478144</td>
      <td>0.497400</td>
      <td>0.909657</td>
      <td>1.252553</td>
      <td>1.273724</td>
      <td>0.392907</td>
      <td>1.153000</td>
      <td>1.295239</td>
    </tr>
    <tr>
      <th>ZMYND10</th>
      <td>0.090382</td>
      <td>0.008073</td>
      <td>0.064288</td>
      <td>0.010513</td>
      <td>0.078923</td>
      <td>0.079593</td>
      <td>0.048360</td>
      <td>0.004594</td>
      <td>0.007390</td>
      <td>0.048804</td>
    </tr>
    <tr>
      <th>ZNF276</th>
      <td>0.193336</td>
      <td>0.240683</td>
      <td>0.235915</td>
      <td>0.076893</td>
      <td>0.039753</td>
      <td>0.243278</td>
      <td>0.412919</td>
      <td>0.118093</td>
      <td>0.164051</td>
      <td>0.218176</td>
    </tr>
    <tr>
      <th>ZNF486</th>
      <td>0.958187</td>
      <td>0.108233</td>
      <td>0.005357</td>
      <td>0.113036</td>
      <td>0.223665</td>
      <td>0.002076</td>
      <td>0.087296</td>
      <td>0.172716</td>
      <td>0.010346</td>
      <td>0.007952</td>
    </tr>
    <tr>
      <th>ZNF683</th>
      <td>0.060481</td>
      <td>0.077736</td>
      <td>0.009184</td>
      <td>0.014918</td>
      <td>0.009756</td>
      <td>0.024224</td>
      <td>0.036456</td>
      <td>0.018888</td>
      <td>0.009079</td>
      <td>0.024949</td>
    </tr>
  </tbody>
</table>
<p>548 rows × 10 columns</p>
</div>



## Setting the parameters for OncoBLADE
Before running OncoBLADE there is one more thing to do, setting the (hyper)parameters.
The key parameters used in OncoBLADE are:
- Hyperparameters (`hyperpars`): `Alpha0`, `Kappa0` and `SigmaY`.
- `Alpha`: Inital guess of `Alpha`. (used to be Hyperparameter in BLADE)
- `Nrep`: Number of repeated optimizations with different initial guess.
- `IterMax`: Maximum number of iterations between variational parameter optimization and updating hyperparameter `Alpha`.
- `Njob`: Number of parallel jobs.



```python
pars = {
    'Alpha': 1,
    'Alpha0': 1000,
    'Kappa0': 1,
    'SY': 1,
    'Nrep': 3, ## small for demo purposes, for real application >10 is advised
    'IterMax': 100, ## small for demo purposes, for real application >1000 is advised
    'Njob': 4 ## Number of parallel jobs.
}
```

## 1) Cell fraction estimation by OncoBLADE
<div>
<img src="Demo_Figures/OncoBLADE_Model.png" width="500"/>
</div>

Now we are ready to run the first step of OncoBLADE, the cell fraction estimation. Here it is applied to 10 NSCLC samples.

OncoBLADE produce several outcomes:
- `final_obj`: final OncoBLADE object with optimized variational/hyperparameters.
- `ELBO`: The objective function value (ELBO function) with the optimized variational/hyperparameters.
- `outs`: Outcome of BLADE for every possible combination of hyperparameters, used in the Empirical Bayes framework.
- `args`: arguments used to run `Framework_Iterative`.



```python
final_obj, ELBO, outs, args = Framework_Iterative(
            Mu_AutoGeneS.to_numpy(), Omega_AutoGeneS.to_numpy(),Y_AutoGeneS.to_numpy(),
            Alpha=pars['Alpha'], Alpha0=pars['Alpha0'], 
            Kappa0=pars['Kappa0'], sY = pars['SY'],
            Nrep= pars['Nrep'], Njob= pars['Njob'], IterMax= pars['IterMax'], Expectation = Expectation)

## Save BLADE output in one dictionary
OncoBLADE_output = {
    'final_obj': final_obj,
    'ELBO': ELBO,
    'outs' : args,
    'genes' : common_AutoGeneS # Autogene selected genes
     }

```

    all of 548 genes are used for optimization.
    Initialization with Support vector regression


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   5 out of  10 | elapsed:    1.3s remaining:    1.3s
    [Parallel(n_jobs=4)]: Done   7 out of  10 | elapsed:    1.6s remaining:    0.7s
    [Parallel(n_jobs=4)]: Done  10 out of  10 | elapsed:    1.9s finished
    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:871: Warning: No optimization is not done yet
      obj.Check_health()
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:871: Warning: No optimization is not done yet
      obj.Check_health()
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:871: Warning: No optimization is not done yet
      obj.Check_health()


    No feature filtering is done (fsel = 0)


    [Parallel(n_jobs=4)]: Done   3 out of   3 | elapsed: 12.0min finished


## 2) Cell type specific gene expression estimation by OncoBLADE
<div>
<img src="Demo_Figures/Further_Optimization.png" width="300"/>
</div>
After step 1, we can fix the estimated cell fractions which allows us to estimate the cell type specific gene expression in parallel per gene, which is feasible for a large amount of genes, $J_{all}$. In this step, the model is allowed to deviate from the cell type-specific gene expression signature and thus taking up residual gene expression levels not explained by the signature, coming from interpatient heterogeneity. Here we estimate the cell type specific gene expression levels for the top 1819 highly variable genes in the scRNAseq data.



```python
## Load autogenes which was saved in the signature
hvgenes = signature['GeneList_hvg']

## Find common hvg with the bulk data
common_hvgenes = [value for value in hvgenes if value in Y.index.to_list()]

## Subset Mu, Omega & bulk on AutoGeneS
Mu_hvgenes = Mu.loc[common_hvgenes,]
Omega_hvgenes = Omega.loc[common_hvgenes,]
Y_hvgenes = Y.loc[common_hvgenes,]

## Estimate cell type specific gene expression
final_obj_2, obj_func = Purify_AllGenes(OncoBLADE_output, Mu_hvgenes.to_numpy(), Omega_hvgenes.to_numpy(), 
                                        Y_hvgenes.to_numpy(), pars['Njob'])


```

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:974: Warning: No optimization is not done yet
      obj.Check_health()
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:974: Warning: No optimization is not done yet
      obj.Check_health()
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:974: Warning: No optimization is not done yet
      obj.Check_health()
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:974: Warning: No optimization is not done yet
      obj.Check_health()
    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    1.7s
    [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    2.7s
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:    4.5s
    [Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:    6.6s
    [Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:    9.2s
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   12.8s
    [Parallel(n_jobs=4)]: Done  53 tasks      | elapsed:   16.1s
    [Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:   19.7s
    [Parallel(n_jobs=4)]: Done  77 tasks      | elapsed:   23.2s
    /home/anoyaro/data/OncoBLADE/Deconvolution/OncoBLADE.py:974: Warning: No optimization is not done yet
      obj.Check_health()
    [Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:   26.4s
    [Parallel(n_jobs=4)]: Done 105 tasks      | elapsed:   28.7s
    [Parallel(n_jobs=4)]: Done 120 tasks      | elapsed:   32.4s
    [Parallel(n_jobs=4)]: Done 137 tasks      | elapsed:   36.0s
    [Parallel(n_jobs=4)]: Done 154 tasks      | elapsed:   39.8s
    [Parallel(n_jobs=4)]: Done 173 tasks      | elapsed:   43.5s
    [Parallel(n_jobs=4)]: Done 192 tasks      | elapsed:   49.4s
    [Parallel(n_jobs=4)]: Done 213 tasks      | elapsed:   53.3s
    [Parallel(n_jobs=4)]: Done 234 tasks      | elapsed:   57.4s
    [Parallel(n_jobs=4)]: Done 257 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=4)]: Done 280 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=4)]: Done 305 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=4)]: Done 330 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=4)]: Done 357 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=4)]: Done 384 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=4)]: Done 413 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=4)]: Done 442 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=4)]: Done 473 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=4)]: Done 504 tasks      | elapsed:  2.1min
    [Parallel(n_jobs=4)]: Done 537 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=4)]: Done 570 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=4)]: Done 605 tasks      | elapsed:  2.5min
    [Parallel(n_jobs=4)]: Done 640 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=4)]: Done 677 tasks      | elapsed:  2.7min
    [Parallel(n_jobs=4)]: Done 714 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=4)]: Done 753 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=4)]: Done 792 tasks      | elapsed:  3.2min
    [Parallel(n_jobs=4)]: Done 833 tasks      | elapsed:  3.3min
    [Parallel(n_jobs=4)]: Done 874 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=4)]: Done 917 tasks      | elapsed:  3.6min
    [Parallel(n_jobs=4)]: Done 960 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=4)]: Done 1005 tasks      | elapsed:  4.0min
    [Parallel(n_jobs=4)]: Done 1050 tasks      | elapsed:  4.1min
    [Parallel(n_jobs=4)]: Done 1097 tasks      | elapsed:  4.3min
    [Parallel(n_jobs=4)]: Done 1144 tasks      | elapsed:  4.5min
    [Parallel(n_jobs=4)]: Done 1193 tasks      | elapsed:  4.7min
    [Parallel(n_jobs=4)]: Done 1242 tasks      | elapsed:  4.9min
    [Parallel(n_jobs=4)]: Done 1293 tasks      | elapsed:  5.1min
    [Parallel(n_jobs=4)]: Done 1344 tasks      | elapsed:  5.3min
    [Parallel(n_jobs=4)]: Done 1397 tasks      | elapsed:  5.4min
    [Parallel(n_jobs=4)]: Done 1450 tasks      | elapsed:  5.6min
    [Parallel(n_jobs=4)]: Done 1505 tasks      | elapsed:  5.8min
    [Parallel(n_jobs=4)]: Done 1560 tasks      | elapsed:  6.0min
    [Parallel(n_jobs=4)]: Done 1617 tasks      | elapsed:  6.2min
    [Parallel(n_jobs=4)]: Done 1674 tasks      | elapsed:  6.4min
    [Parallel(n_jobs=4)]: Done 1733 tasks      | elapsed:  6.6min
    [Parallel(n_jobs=4)]: Done 1792 tasks      | elapsed:  6.8min
    [Parallel(n_jobs=4)]: Done 1819 out of 1819 | elapsed:  6.9min finished


## 3) Downstream analysis
Next we will show how to extract the posterior estimates of the fractions ($\hat{f}_{i}^{t}$) and cell type-specifc gene expression profiles ($\hat{x}_{ij}^{t}$) and perform some downstream analysis


```python
# Fetch posterior estimates of cell fractions
Fractions = pd.DataFrame(final_obj.ExpF(final_obj.Beta),index = bulk.set_index('symbol').columns, columns = signature['celltype_list'])
Fractions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cancer_cell</th>
      <th>Lung_endothelial_cell</th>
      <th>Fibroblast</th>
      <th>Macrophage</th>
      <th>Plasma_cell</th>
      <th>Lung_epithelial_cell</th>
      <th>CD4_proliferating_T_cell</th>
      <th>Monocyte</th>
      <th>Mast_cell</th>
      <th>DC</th>
      <th>CD8_effector_T_cell</th>
      <th>B_cell</th>
      <th>Neutrophil</th>
      <th>Alveolar_cell</th>
      <th>CD4_Th17_like_cell</th>
      <th>CD4_Treg</th>
      <th>NK_cell</th>
      <th>CD4_naive_T_cell</th>
      <th>CD8_exhausted_T_cell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-91-6840</th>
      <td>0.457850</td>
      <td>0.054249</td>
      <td>0.073034</td>
      <td>0.033775</td>
      <td>0.037359</td>
      <td>0.018281</td>
      <td>0.023344</td>
      <td>0.013747</td>
      <td>0.016594</td>
      <td>0.026039</td>
      <td>0.025426</td>
      <td>0.031764</td>
      <td>0.006164</td>
      <td>0.066738</td>
      <td>0.028317</td>
      <td>0.025492</td>
      <td>0.014848</td>
      <td>0.027555</td>
      <td>0.019424</td>
    </tr>
    <tr>
      <th>TCGA-78-8655</th>
      <td>0.439525</td>
      <td>0.047632</td>
      <td>0.069957</td>
      <td>0.054368</td>
      <td>0.033866</td>
      <td>0.018504</td>
      <td>0.020229</td>
      <td>0.017029</td>
      <td>0.017366</td>
      <td>0.032719</td>
      <td>0.024888</td>
      <td>0.028862</td>
      <td>0.006385</td>
      <td>0.075612</td>
      <td>0.027395</td>
      <td>0.024846</td>
      <td>0.015096</td>
      <td>0.026595</td>
      <td>0.019125</td>
    </tr>
    <tr>
      <th>TCGA-NJ-A7XG</th>
      <td>0.397684</td>
      <td>0.041284</td>
      <td>0.060115</td>
      <td>0.027990</td>
      <td>0.039414</td>
      <td>0.023248</td>
      <td>0.021460</td>
      <td>0.013410</td>
      <td>0.019218</td>
      <td>0.025511</td>
      <td>0.032264</td>
      <td>0.036782</td>
      <td>0.004913</td>
      <td>0.123387</td>
      <td>0.033555</td>
      <td>0.030993</td>
      <td>0.013679</td>
      <td>0.033570</td>
      <td>0.021524</td>
    </tr>
    <tr>
      <th>TCGA-38-4625</th>
      <td>0.246393</td>
      <td>0.055329</td>
      <td>0.094822</td>
      <td>0.075323</td>
      <td>0.053766</td>
      <td>0.021873</td>
      <td>0.039029</td>
      <td>0.020572</td>
      <td>0.021541</td>
      <td>0.043028</td>
      <td>0.035022</td>
      <td>0.043535</td>
      <td>0.007221</td>
      <td>0.084887</td>
      <td>0.039897</td>
      <td>0.034737</td>
      <td>0.018822</td>
      <td>0.038594</td>
      <td>0.025607</td>
    </tr>
    <tr>
      <th>TCGA-38-4627</th>
      <td>0.196398</td>
      <td>0.082762</td>
      <td>0.180566</td>
      <td>0.078308</td>
      <td>0.044615</td>
      <td>0.025505</td>
      <td>0.023089</td>
      <td>0.023734</td>
      <td>0.021604</td>
      <td>0.046746</td>
      <td>0.029805</td>
      <td>0.036523</td>
      <td>0.005524</td>
      <td>0.071599</td>
      <td>0.033053</td>
      <td>0.030443</td>
      <td>0.014527</td>
      <td>0.032038</td>
      <td>0.023162</td>
    </tr>
    <tr>
      <th>TCGA-85-A4QQ</th>
      <td>0.382782</td>
      <td>0.046996</td>
      <td>0.074997</td>
      <td>0.032322</td>
      <td>0.048026</td>
      <td>0.023717</td>
      <td>0.031926</td>
      <td>0.013522</td>
      <td>0.017940</td>
      <td>0.028436</td>
      <td>0.036164</td>
      <td>0.040724</td>
      <td>0.004989</td>
      <td>0.065219</td>
      <td>0.041110</td>
      <td>0.034353</td>
      <td>0.014267</td>
      <td>0.039055</td>
      <td>0.023454</td>
    </tr>
    <tr>
      <th>TCGA-56-8628</th>
      <td>0.368941</td>
      <td>0.060837</td>
      <td>0.101053</td>
      <td>0.062645</td>
      <td>0.036514</td>
      <td>0.021965</td>
      <td>0.022252</td>
      <td>0.018642</td>
      <td>0.015908</td>
      <td>0.039000</td>
      <td>0.025225</td>
      <td>0.029880</td>
      <td>0.005030</td>
      <td>0.074609</td>
      <td>0.027954</td>
      <td>0.026569</td>
      <td>0.012818</td>
      <td>0.028340</td>
      <td>0.021818</td>
    </tr>
    <tr>
      <th>TCGA-63-A5M9</th>
      <td>0.552290</td>
      <td>0.035306</td>
      <td>0.063609</td>
      <td>0.027675</td>
      <td>0.030677</td>
      <td>0.015804</td>
      <td>0.023961</td>
      <td>0.014380</td>
      <td>0.014663</td>
      <td>0.022731</td>
      <td>0.022424</td>
      <td>0.025561</td>
      <td>0.005286</td>
      <td>0.042535</td>
      <td>0.024601</td>
      <td>0.022409</td>
      <td>0.014512</td>
      <td>0.024021</td>
      <td>0.017555</td>
    </tr>
    <tr>
      <th>TCGA-77-8128</th>
      <td>0.439615</td>
      <td>0.046968</td>
      <td>0.102643</td>
      <td>0.027347</td>
      <td>0.037960</td>
      <td>0.018016</td>
      <td>0.029760</td>
      <td>0.013194</td>
      <td>0.015846</td>
      <td>0.023901</td>
      <td>0.028987</td>
      <td>0.033113</td>
      <td>0.005417</td>
      <td>0.050540</td>
      <td>0.031943</td>
      <td>0.028435</td>
      <td>0.013825</td>
      <td>0.031529</td>
      <td>0.020960</td>
    </tr>
    <tr>
      <th>TCGA-34-5231</th>
      <td>0.437839</td>
      <td>0.044467</td>
      <td>0.075098</td>
      <td>0.031077</td>
      <td>0.043515</td>
      <td>0.018084</td>
      <td>0.035941</td>
      <td>0.014117</td>
      <td>0.015571</td>
      <td>0.026207</td>
      <td>0.029625</td>
      <td>0.034584</td>
      <td>0.005797</td>
      <td>0.054587</td>
      <td>0.033072</td>
      <td>0.030412</td>
      <td>0.014634</td>
      <td>0.032682</td>
      <td>0.022691</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fetch posterior estimates of cell type specific gene expression profiles from malignant cells
Malignant_cell_profiles = pd.DataFrame(final_obj_2.Nu[:,:,signature['celltype_list'].index('Cancer_cell')], index = bulk.set_index('symbol').columns, columns = common_hvgenes )
Malignant_cell_profiles
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A2M</th>
      <th>AARD</th>
      <th>AATK</th>
      <th>ABCA1</th>
      <th>ABCA3</th>
      <th>ABCA7</th>
      <th>ABCA8</th>
      <th>ABCB1</th>
      <th>ABCC9</th>
      <th>ABHD5</th>
      <th>...</th>
      <th>ZNF276</th>
      <th>ZNF292</th>
      <th>ZNF331</th>
      <th>ZNF429</th>
      <th>ZNF438</th>
      <th>ZNF486</th>
      <th>ZNF683</th>
      <th>ZNF821</th>
      <th>ZNF831</th>
      <th>ZNF90</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-91-6840</th>
      <td>0.103638</td>
      <td>0.196936</td>
      <td>0.065189</td>
      <td>0.753481</td>
      <td>0.418027</td>
      <td>0.222626</td>
      <td>0.010644</td>
      <td>0.007029</td>
      <td>0.050494</td>
      <td>0.094724</td>
      <td>...</td>
      <td>0.236436</td>
      <td>0.471539</td>
      <td>0.252608</td>
      <td>0.208905</td>
      <td>0.091383</td>
      <td>0.856317</td>
      <td>0.030870</td>
      <td>0.050046</td>
      <td>0.010064</td>
      <td>0.359194</td>
    </tr>
    <tr>
      <th>TCGA-78-8655</th>
      <td>0.103130</td>
      <td>0.033365</td>
      <td>0.058693</td>
      <td>0.507224</td>
      <td>0.508957</td>
      <td>0.395935</td>
      <td>0.012140</td>
      <td>0.002422</td>
      <td>0.042937</td>
      <td>0.221377</td>
      <td>...</td>
      <td>0.289310</td>
      <td>0.186407</td>
      <td>0.169424</td>
      <td>0.110066</td>
      <td>0.124586</td>
      <td>0.200032</td>
      <td>0.034034</td>
      <td>0.074779</td>
      <td>0.007454</td>
      <td>0.047940</td>
    </tr>
    <tr>
      <th>TCGA-NJ-A7XG</th>
      <td>0.107512</td>
      <td>0.033037</td>
      <td>0.065768</td>
      <td>0.252096</td>
      <td>0.443706</td>
      <td>0.709901</td>
      <td>0.009415</td>
      <td>0.000409</td>
      <td>0.026941</td>
      <td>0.118116</td>
      <td>...</td>
      <td>0.284145</td>
      <td>0.347346</td>
      <td>0.250575</td>
      <td>0.105566</td>
      <td>0.041722</td>
      <td>0.104249</td>
      <td>0.020199</td>
      <td>0.082562</td>
      <td>0.004360</td>
      <td>0.050830</td>
    </tr>
    <tr>
      <th>TCGA-38-4625</th>
      <td>0.102140</td>
      <td>0.043803</td>
      <td>0.053211</td>
      <td>0.967671</td>
      <td>0.479317</td>
      <td>0.448948</td>
      <td>0.010448</td>
      <td>0.002763</td>
      <td>0.039594</td>
      <td>0.379105</td>
      <td>...</td>
      <td>0.154254</td>
      <td>0.343194</td>
      <td>0.242947</td>
      <td>0.090785</td>
      <td>0.119034</td>
      <td>0.246241</td>
      <td>0.021887</td>
      <td>0.007003</td>
      <td>0.006085</td>
      <td>0.099909</td>
    </tr>
    <tr>
      <th>TCGA-38-4627</th>
      <td>0.099113</td>
      <td>0.042416</td>
      <td>0.051366</td>
      <td>0.541478</td>
      <td>0.409586</td>
      <td>0.324958</td>
      <td>0.012037</td>
      <td>0.006669</td>
      <td>0.046520</td>
      <td>0.204532</td>
      <td>...</td>
      <td>0.141969</td>
      <td>0.259905</td>
      <td>0.225018</td>
      <td>0.067863</td>
      <td>0.108536</td>
      <td>0.346748</td>
      <td>0.021354</td>
      <td>0.060049</td>
      <td>0.005748</td>
      <td>0.097078</td>
    </tr>
    <tr>
      <th>TCGA-85-A4QQ</th>
      <td>0.098730</td>
      <td>0.029081</td>
      <td>0.047214</td>
      <td>0.635764</td>
      <td>0.271487</td>
      <td>0.361020</td>
      <td>0.008132</td>
      <td>0.000887</td>
      <td>0.024608</td>
      <td>0.159796</td>
      <td>...</td>
      <td>0.299349</td>
      <td>0.346821</td>
      <td>0.206850</td>
      <td>0.015718</td>
      <td>0.050300</td>
      <td>0.101577</td>
      <td>0.022755</td>
      <td>0.003736</td>
      <td>0.004240</td>
      <td>0.061269</td>
    </tr>
    <tr>
      <th>TCGA-56-8628</th>
      <td>0.100490</td>
      <td>0.037937</td>
      <td>0.080949</td>
      <td>0.716933</td>
      <td>0.459477</td>
      <td>0.479628</td>
      <td>0.015903</td>
      <td>0.004291</td>
      <td>0.044587</td>
      <td>0.090211</td>
      <td>...</td>
      <td>0.476041</td>
      <td>0.630135</td>
      <td>0.266294</td>
      <td>0.094929</td>
      <td>0.083121</td>
      <td>0.195447</td>
      <td>0.025222</td>
      <td>0.070242</td>
      <td>0.009217</td>
      <td>0.081399</td>
    </tr>
    <tr>
      <th>TCGA-63-A5M9</th>
      <td>0.113275</td>
      <td>0.070545</td>
      <td>0.041826</td>
      <td>0.276080</td>
      <td>0.381818</td>
      <td>0.291636</td>
      <td>0.009688</td>
      <td>0.003894</td>
      <td>0.031490</td>
      <td>0.130872</td>
      <td>...</td>
      <td>0.142084</td>
      <td>0.893575</td>
      <td>0.299995</td>
      <td>0.059357</td>
      <td>0.108009</td>
      <td>0.241000</td>
      <td>0.021276</td>
      <td>0.058894</td>
      <td>0.006574</td>
      <td>0.114311</td>
    </tr>
    <tr>
      <th>TCGA-77-8128</th>
      <td>0.105476</td>
      <td>0.024161</td>
      <td>0.044704</td>
      <td>0.773183</td>
      <td>0.352193</td>
      <td>0.291296</td>
      <td>0.007449</td>
      <td>0.000611</td>
      <td>0.031182</td>
      <td>0.234786</td>
      <td>...</td>
      <td>0.210151</td>
      <td>0.590495</td>
      <td>0.175070</td>
      <td>0.042336</td>
      <td>0.034323</td>
      <td>0.096719</td>
      <td>0.019822</td>
      <td>0.051980</td>
      <td>0.004115</td>
      <td>0.052390</td>
    </tr>
    <tr>
      <th>TCGA-34-5231</th>
      <td>0.106897</td>
      <td>0.026950</td>
      <td>0.064846</td>
      <td>0.245757</td>
      <td>0.350117</td>
      <td>0.502846</td>
      <td>0.009308</td>
      <td>0.004358</td>
      <td>0.038476</td>
      <td>0.196769</td>
      <td>...</td>
      <td>0.267012</td>
      <td>0.336968</td>
      <td>0.139254</td>
      <td>0.083087</td>
      <td>0.092408</td>
      <td>0.094858</td>
      <td>0.023057</td>
      <td>0.023648</td>
      <td>0.006005</td>
      <td>0.048323</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 1819 columns</p>
</div>



The bulk dataset contains 5 lung tumors with an adenocarcinoma histology (LUAD) and 5 with an squamous cell histology (LUSC).
In the next section we perform unsupervised clustering analysis to investigate whether there are differences between the two histological subtypes in terms of fractions.


```python
# add color annotation for LUAD/LUSC: First 5 samples are LUAD and the second 5 are LUSC
Subtype_colors = ['red']*5+['green']*5
# plot clustermap
map = sns.clustermap(zscore(Fractions).transpose(), figsize = (6,6),col_colors=Subtype_colors)
# add color legend
plt.legend([Patch(facecolor=col) for col in ['red','green']], ['LUAD','LUSC'],
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right',frameon=False)
```




    <matplotlib.legend.Legend at 0x7f3febb55090>




    
![png](OncoBLADE_Demo_files/OncoBLADE_Demo_21_1.png)
    


Since the histological subtype should mainly concern the malignant cells, difference in their gene expression profiles should be reflected in the estimated cell type-specific gene expression profiles.
Therefore, when clustering these malignant profiles, a split between LUAD an LUSC is expected.



```python
# plot clustermap
map = sns.clustermap(zscore(Malignant_cell_profiles).transpose(), figsize = (5,5),col_colors=Subtype_colors, metric = 'correlation')
# add color legend
plt.legend([Patch(facecolor=col) for col in ['red','green']], ['LUAD','LUSC'],
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right',frameon=False)
```

    /home/anoyaro/anaconda3/envs/OncoBLADE/lib/python3.10/site-packages/seaborn/matrix.py:560: UserWarning: Clustering large matrix with scipy. Installing `fastcluster` may give better performance.
      warnings.warn(msg)
    /home/anoyaro/anaconda3/envs/OncoBLADE/lib/python3.10/site-packages/seaborn/matrix.py:560: UserWarning: Clustering large matrix with scipy. Installing `fastcluster` may give better performance.
      warnings.warn(msg)





    <matplotlib.legend.Legend at 0x7f3febcaffa0>




    
![png](OncoBLADE_Demo_files/OncoBLADE_Demo_23_2.png)
    

