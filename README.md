## Ensembling Language Models to Improve Robustness to Spurious Correlations

Group Members: Allen Cheung, Yong-Won Cho, Maya Deshpande, Melinda Ma

In this work, we investigate the effectiveness of ensembling various language models with different pre-training objectives to improve their robustness to spurious correlations. Specifically, we fine-tune pre-trained BERT, ALBERT, and RoBERTa models on a large dataset for the downstream task of natural language inference. As a baseline for comparison, we evaluate their individual test accuracies on two test sets designed to study spurious correlations. Using a weighted majority voting ensembling approach, we ensemble different combinations of the models and compare them with the baselines.

- Part of code (fine-tuning) is derived from `https://github.com/dh1105/Sentence-Entailment/tree/main`

- Negation labels is taken from `https://github.com/kohpangwei/group_DRO/tree/master/dataset_metadata/multinli`