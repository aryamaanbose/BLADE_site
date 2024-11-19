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
          <header className={styles.heroBanner}>
            <h1 className={styles.title}>BLADE (Bayesian Log Normal Deconvolution)</h1>
            <p className={styles.tagline}>
              <strong>BLADE</strong> was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single step, accounting for observed gene expression variability in single-cell RNA-seq data.
            </p>
          </header>

          <section className={styles.descriptionSection}>
            <p>
              <strong>BLADE Framework:</strong> To construct prior knowledge, we used single-cell sequencing data. Cells undergo phenotyping, clustering, and differential gene expression analysis. For each cell type, we retrieve average expression profiles (red cross and top heatmap) and standard deviation per gene (blue circle and bottom heatmap). This prior knowledge is then utilized in the hierarchical Bayesian model (bottom right) to deconvolute bulk gene expression data.
            </p>
          </section>

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
