---
title: 'ipfml: A Python package for `Image Processing For Machine Learning`'
tags:
  - Python
  - Image Processing
  - Computer Graphics
  - Statistics
  - Global illumination
authors:
  - name: Jérôme BUISINE
    orcid: 0000-0001-6071-744X
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Univ. Littoral Côte d’Opale, LISIC Calais, France, F-62100
   index: 1
date: 9 June 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

# Motivation

`ipfml` Python package have been developed during thesis project in order to regroup all usefull methods for image processing for machine learning problems.

This Python package is subdivised into multiple modules:

- **metrics:** Metrics computation for model performance (between predicted and known labels)
- **utils:** All utils functions developed (such as normalization of images or data)
- **exceptions:** All customized exceptions
- **filters:** Image filter module with convolution (additive noise using convolution)
- **iqa:** Image quality assessments (full-reference IQA mainly)
- **processing:** Image processing module (reduction dimension, transformation, segmentation...) 

# Application



# Acknowledgements

This work is supported by *Agence Nationale de la Recherche* : project ANR-17-CE38-0009

# References