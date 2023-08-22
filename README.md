# HGCLMDA: Predicting mRNA-drug sensitivity associations via hypergraph contrastive learning
HGCLMDA is noval a hypergraph contrastive learning approach to predict potential mRNA-drug sensitivity associations. HGCLMDA integrates a graph convolutional network-based method with a hypergraph convolutional network to mine high-order relationships between mRNA-drug association pairs. The proposed cross-view contrastive learning architecture improves the model's learning ability, and the inner product is used to obtain the mRNA-drug sensitivity association score.
# Requirements
- torch 1.8.1
- python 3.7.16
- numpy 1.21.6
- pandas 1.3.5
- scikit-learn 1.0.2
- scipy 1.7.3
# Data
RNAactDrug is a comprehensive RNA molecule-drug sensitivity association database. Up to now, RNAactDrug contains more than 4,924,200 RNA molecule-drug sensitivity associations, covering more than 19,770 mRNAs, 11,119 lncRNAs, 438 miRNAs, and 4,155 drugs. We obtained the mRNA-drug sensitivity association data from the RNAactDrug database. 

To verify the performance of HGCLMDA, we constructed three datasets, MDA1, MDA2, and MDA3, according to the p-values score of mRNA and drug less than 0.05, 0.005, and 0.0005, respectively.

In addition, in order to verify the performance of HGCLMDA on sparse datasets, we constructed four sparse datasets. We divided four new datasets, MDA_20, MDA_30, MDA_50, and MDA_60, according to the number of mRNA-drug association pairs 10-20, 20-30, 40-50 and 50-60 in the MDA1 dataset.

# Run the demo
```
python HGCLMDA.py
```
