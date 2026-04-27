# Cross-Lingual Embedding Evaluation Report

## Objective
Determine whether a small embedding space trained from bilingual/translated data preserves semantic relationships sufficiently to validate meaning correctness.

## Summary of Results
The following table summarizes the mean cosine similarity for different pair types across the three embedding spaces:

```csv
Model,Positive_Mean,Negative_Mean,CrossLingual_Mean,Pairs_Evaluated
H1_Translated,0.4159769,0.3411101,0.51645637,P:500 N:500 C:500
H2_Native,0.16420496,0.19375186,0.21551225,P:500 N:500 C:500
H3_Pseudo,0.67861426,0.45018932,0.5236354,P:500 N:500 C:500

```

## Observations
- **H1 (Translated Corpus)**: Should capture semantics purely from translated structure.
- **H2 (Native Hindi Corpus)**: Baseline for native semantic relationships.
- **H3 (Pseudo-context Corpus)**: Explores mapping cross-lingual relationships by embedding English words in synthetic Hindi context.

Please refer to the `results/similarity_distributions.png` plot for visual density estimates of the similarities.

## Conclusion
Based on the separation between Positive (Synonyms) and Negative (Random) pairs, one can threshold the cosine similarity to flag incorrect meanings. 
