import React from 'react';
import Layout from '@theme/Layout';
import styles from './index.module.css';

export default function Home() {
  return (
    <Layout
      title="BLADE"
      description="BLADE - Bayesian Log Normal Deconvolution"
    >
      <main>
        <div className="container">
          <h1 className={styles.title}>
            BLADE (Bayesian Log Normal Deconvolution)
          </h1>
          <p>
            <strong>BLADE</strong> was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single-step while accounting for the observed gene expression variability in single-cell RNA-seq data.
          </p>
          <p>
            <strong>BLADE framework.</strong> To construct a prior knowledge of BLADE, we used single-cell sequencing data. Cells are subject to phenotyping, clustering, and differential gene expression analysis. Then, for each cell type, we retrieve average expression profiles (red cross and top heatmap), and standard deviation per gene (blue circle and bottom heatmap). This prior knowledge is then used in the hierarchical Bayesian model (bottom right) to deconvolute bulk gene expression data.
          </p>
          <div className={styles.imageContainer}>
            <img
              src="/img/framework.png"
              alt="BLADE Framework"
              className={styles.responsiveImage}
            />
          </div>
        </div>
      </main>
    </Layout>
  );
}
