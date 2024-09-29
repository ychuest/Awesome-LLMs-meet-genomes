[python-img]: https://img.shields.io/github/languages/top/ychuest/Awesome-LLMs-meet-genomes?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/ychuest/Awesome-LLMs-meet-genomes?color=yellow
[stars-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes/stargazers
[fork-img]: https://img.shields.io/github/forks/ychuest/Awesome-LLMs-meet-genomes?color=lightblue&label=fork
[fork-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes/network/members
[visitors-img]: https://profile-counter.glitch.me/Awesome-LLMs-meet-genomes/count.svg
[adgc-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes


# Awesome-LLMs-meet-genomes

Awesome-LLMs-meet-genomes is a collection of state-of-the-art, novel, exciting LLMs methods on genomes.  It contains papers, codes, datasets, evaluations, and analyses. Any additional information about LLMs for bioinformatics is welcome, and we are glad to add you to the contributor list [here](#contributors). Any problems, please contact yangchengyjs@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles:

-----------------------------------
[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=ychuest.Awesome-LLMs-meet-genomes)](https://github.com/ychuest/Awesome-LLMs-meet-genomes)

<p align="center"> 
  Visitor counts<br>
  <img src="https://profile-counter.glitch.me/Awesome-LLMs-meet-genomes/count.svg" />
</p>

## Table of Content
- [Awesome-LLMs-meet-genomes](#awesome-llms-meet-genomes)
  - [Table of Content](#table-of-content)
  - [🔔 News](#-news)
  - [Important Survey Papers](#important-survey-papers)
  - [Genomic Large Language Models (Gene-LLMs)](#genomic-large-language-models-gene-llms)
    - [Generic Base Models](#generic-base-models)
    - [Downstream Tasks](#downstream-tasks)
      - [Retrieval-Augmented Generation](#retrieval-augmented-generation)
      - [Function Prediction](#function-prediction)
      - [Perturbation](#perturbation)
      - [**Variants and Evolution Prediction**](#variants-and-evolution-prediction)
      - [Fine-tuning for Genomes and proteins](#fine-tuning-for-genomes-and-proteins)
      - [Interaction Prediction](#interaction-prediction)
      - [Identification of Transcription Factor Binding Sites](#identification-of-transcription-factor-binding-sites)
      - [Origins of Replication Rite Prediction](#origins-of-replication-rite-prediction)
      - [DNA-binding Protein Prediction](#dna-binding-protein-prediction)
      - [RNA Prediction](#rna-prediction)
      - [**Sequence Modeling**](#sequence-modeling)
  - [Basics of Sequence Modeling](#basics-of-sequence-modeling)
  - [Quantization](#quantization)
  - [Fine-tuning](#fine-tuning)
  - [Reducing Knowledge Hallucination](#reducing-knowledge-hallucination)
  - [Other Related Awesome Repository](#other-related-awesome-repository)
  - [Contributors](#contributors)
---

## 🔔 News

- 🧬✔️ [2024/09] **Benchmarks for classification of genomic sequences** [link](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks).
- 💥 [2024/08] Some real-world experience in training LLMs [link](https://mp.weixin.qq.com/s/ItpCTCcMjTWQJtgpvdwTfw).
- 💥 [2024/08] Three ways of Fine-tuning [link](https://mp.weixin.qq.com/s/MWRW6zZKbK1xJRNsNDOquA).
- 💥 [2024/08] Visualisation of the Transformer Principle [link](https://mp.weixin.qq.com/s/7RBWwf4bQF31E0BySeSx9g).
- 🌟 [2024/08] The Cultivation Method of Large Language Models: A Path to Success [link](https://github.com/wdndev/llm_interview_note).
- 📖 [2024/08] Large Language Models: From Theory to Practice [link](./docs/1.pdf).


## Important Survey Papers


| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2024.07 | **Genomic Language Models: Opportunities and Challenges** |    arXiv   | [Link](https://arxiv.org/abs/2407.11435) |  - |
| 2024.07 | **Scientific Large Language Models: A Survey on Biological & Chemical Domains** |    arXiv   | [Link](https://arxiv.org/abs/2401.14656) |  [link](https://github.com/HICAI-ZJU/Scientific-LLM-Survey) |
| 2024.01 | **Large language models in bioinformatics: applications and perspectives** |    arXiv    | [Link](https://arxiv.org/abs/2401.04155v1) |  - |
| 2023.11 | **To Transformers and Beyond: Large Language Models for the Genome** |    arXiv    | [Link](https://arxiv.org/abs/2311.07621) |  -   |
| 2023.01 | **Applications of transformer-based language models in bioinformatics: a survey** |    Bioinformatics Advances    | [Link](https://arxiv.org/abs/2311.07621) |  -   |





## Genomic Large Language Models (Gene-LLMs)



### Generic Base Models
| Year | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **dnaGrinder: a lightweight and high-capacity genomic foundation model** |    arXiv   | [link](https://doi.org/10.48550/arXiv.2409.15697)  |                              -                               |
| 2024.08 | **Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision** |    Bioinformatics Advances   | [link](https://doi.org/10.1093/bioadv/vbae117)  |                              [link](https://github.itap.purdue.edu/Clan-labs/ENBED)                               |
| 2024.08 | **Unlocking Efficiency: Adaptive Masking for Gene Transformer Models** |  ECAI'24    | [link](https://arxiv.org/pdf/2408.07180)  |                              [link](https://github.com/roysoumya/curriculum-GeneMask)                           |
| 2024.07 ✨✨✨| **VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling** |  ICML'24    | [link](https://arxiv.org/pdf/2405.10812)  |                              [link](https://github.com/Lupin1998/VQDNA)                           |
| 2024.07 | **OmniGenome: Aligning RNA Sequences with Secondary Structures in Genomic Foundation Models** |  arXiv    | [link](https://arxiv.org/abs/2407.11242)  |                              [link](https://github.com/yangheng95/OmniGenomeBench)                           |
| 2024.07 | **Scorpio : Enhancing Embeddings to Improve Downstream Analysis of DNA sequences** |  bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.07.19.604359v1.abstract)  |                              [link](https://github.com/EESI/Scorpio)                            |
| 2024.07 | **DNA language model GROVER learns sequence context in the human genome (可用于蛋白质-DNA结合预测任务)** |   Nature Machine Intelligence    | [link](https://doi.org/10.1038/s42256-024-00872-0)  |                              [link](https://doi.org/10.5281/zenodo.8373202)   [tutorials](https://doi.org/10.5281/zenodo.8373158)                            |
| 2024.05 | **Are Genomic Language Models All You Need? Exploring Genomic Language Models on Protein Downstream Tasks** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.05.20.594989v1)  |                              [link](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species)                           |
| 2024.05 | **GeneAgent: Self-verification Language Agent for Gene Set Knowledge Discovery using Domain Databases** |   arXiv    | [link](https://arxiv.org/pdf/2405.16205)  |              -                       |
| 2024.05 | **Self-Distillation Improves DNA Sequence Inference Databases** |   arXiv    | [link](https://arxiv.org/pdf/2405.08538)  |              [link](https://github.com/wiedersehne/FinDNA)                       |
| 2024.04 | **Effect of tokenization on transformers for biological sequences** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae196)  |                              [link](https://github.com/technion-cs-nlp/BiologicalTokenizers)                               |
| 2024.04 | **DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome** |   ICLR'24    | [link](https://openreview.net/pdf?id=oMLQB4EZE1)  |                              [link](https://github.com/MAGICS-LAB/DNABERT_2)                               |
| 2024.02 | **Exploring Genomic Large Language Models: Bridging the Gap between Natural Language and Gene Sequences** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.26.581496v1)  |                                            [link](https://github.com/Huatsing-Lau/GenomicLLM)  [data](https://zenodo.org/records/10695802)               |
| 2024.02 | **Sequence modeling and design from molecular to genome scale with Evo** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1)  |                                             [link](https://github.com/evo-design/evo)                |
| 2024.01 | **ProkBERT family: genomic language models for microbiome applications** |    Frontiers in Microbiology    | [Link](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1331233/full) |  [link](https://github.com/nbrgppcu/prokbert)   |
| 2023.09 | **The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3)  |                              [link](https://github.com/instadeepai/nucleotide-transformer)                           |
| 2023.08 | **DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks** |   bioRxiv    | [link](https://www.bioRxiv.org/content/10.1101/2023.07.11.548628v2)  |                                             [link](https://github.com/TencentAILabHealthcare/DNAGPT)                |
| 2023.07 | **EpiGePT: a Pretrained Transformer model for epigenomics** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.07.15.549134v2)  |                              [link](https://github.com/ZjGaothu/EpiGePT)                               |
| 2023.07 | **GeneMask: Fast Pretraining of Gene Sequences to Enable Few-Shot Learning** |   ECAI'23    | [link](https://ebooks.iospress.nl/doi/10.3233/FAIA230492)  |                              [link](https://github.com/roysoumya/genemask)                               
| 2023.06 | **Transfer learning enables predictions in network biology** |   nature   | [link](https://www.nature.com/articles/s41586-023-06139-9)  |                              [link](https://github.com/jkobject/geneformer)                               |
| 2023.06 | **GENA-LM: A Family of Open-Source Foundational DNA Language Models for Long Sequences** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v2.abstract)  | [link](https://github.com/AIRI-Institute/GENA_LM)                               |
| 2023.06 | **HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution** |   NIPS'23    | [link](https://dl.acm.org/doi/10.5555/3666122.3667994)  |                              [link](https://github.com/HazyResearch/hyena-dna)                               |
| 2023.01 | **The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)  |                              [link](https://github.com/instadeepai/nucleotide-transformer)                               |
| 2023.01 | **Species-aware DNA language modeling** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v1)  |            [link](https://github.com/DennisGankin/species-aware-DNA-LM)                                                 |
| 2022.08 | **MoDNA: motif-oriented pre-training for DNA language model** |   BCB'22    | [link](https://dl.acm.org/doi/10.1145/3535508.3545512)  |            [link](https://github.com/uta-smile/MoDNA)                                                |
| 2021.02 | **DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome** |   Bioinformatics    | [link](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)  |                              [link](https://github.com/jerryji1993/DNABERT)                               |


### Downstream Tasks

#### Retrieval-Augmented Generation
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **GeneRAG: Enhancing Large Language Models with Gene-Related Task by Retrieval-Augmented Generation** |  bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.06.24.600176v1.abstract)  |                              [link](https://zenodo.org/records/13119834)                           |

#### Function Prediction

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.07 | **FGBERT: Function-Driven Pre-trained Gene Language Model for Metagenomics** |   arXiv    | [link](https://arxiv.org/abs/2402.16901) |                              -                               |
| 2023.07 | **PLPMpro: Enhancing promoter sequence prediction with prompt-learning based pre-trained language model** |   CIBM    | [link](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007254) |                              -                               |
| 2021.10 | **Effective gene expression prediction from sequence by integrating long-range interactions** |   Nature Methods    | [link](https://www.nature.com/articles/s41592-021-01252-x) |                              [link](https://github.com/deepmind/deepmind-research/tree/master/enformer)                               |


#### Perturbation
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.08 | **Scouter: a transcriptional response predictor for unseen genetic perturbtions with LLM embeddings** |    pypi   | [link](https://pypi.org/project/scouter-learn/) | [link](https://github.com/PancakeZoy/scouter/tree/master) |
| 2024.07 | **Enhancing generative perturbation models with LLM-informed gene embeddings** |    ICLR'24 Workshop   | [link](https://openreview.net/pdf?id=eb3ndUlkt4) | - |
| 2024.03 | **A genome-scale deep learning model to predict gene expression changes of genetic perturbations from multiplex biological networks** |    arXiv   | [link](https://arxiv.org/abs/2403.02724) | [link](https://github.com/lipi12q/TranscriptionNet) |




#### **Variants and Evolution Prediction**

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.04 | **Species-aware DNA language models capture regulatory elements and their evolution** |   Genome Biology    | [link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03221-x)  |            [link](https://github.com/gagneurlab/SpeciesLM) |
| 2023.10 | **GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction** |   bioRxiv    | [link](https://www.bioRxiv.org/content/10.1101/2023.10.10.561776v1.abstract) |                              [link](https://github.com/clinfo/GPN-MSA-env)                              |
| 2023.10 | **GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics** |   The International Journal of High Performance Computing Applications    | [link](https://journals.sagepub.com/doi/10.1177/10943420231201154) |                              [link](https://github.com/ramanathanlab/genslm)                               |
| 2023.08 | **DNA language models are powerful zero-shot predictors of non-coding variant effects** |   arXiv    | [link](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1) |                              [link](https://github.com/songlab-cal/gpn)                               |


#### Fine-tuning for Genomes and proteins
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.08 | **Enhancing recognition and interpretation of functional phenotypic sequences through fine-tuning pre-trained genomic models** |    Journal of Translational Medicine  | [link](https://link.springer.com/article/10.1186/s12967-024-05567-z) | [link](https://github.com/GeorgeBGM/Genome_Fine-Tuning) |
| 2024.08 | **Fine-tuning protein language models boosts predictions across diverse tasks** |    Nature Communications  | [link](https://www.nature.com/articles/s41467-024-51844-2) | [link](https://github.com/whatdoidohaha/RFA) |
| 2024.02 | **Efficient and Scalable Fine-Tune of Language Models for Genome Understanding** |    arXiv  | [link](https://arxiv.org/abs/2402.08075) | [link](https://github.com/zhanglab-aim/LINGO) |
| 2023.11 | **Parameter-Efficient Fine-Tune on Open Pre-trained Transformers for Genomic Sequence** |    NeurIPS'23 Workshop GenBio | [link](https://openreview.net/forum?id=HVQoom7ip2) | - |


#### Interaction Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.08 | **Large-Scale Multi-omic Biosequence Transformers for Modeling Peptide-Nucleotide Interactions** |   arXiv     | [link](https://arxiv.org/abs/2408.16245)  |                               [link](https://github.com/nyuolab/OmniBioTE)                              |
| 2024.04 | **Genomic language model predicts protein co-regulation and function** |   nature communications     | [link](https://www.nature.com/articles/s41467-024-46947-9)  |                               [link](https://github.com/y-hwang/gLM)                              |
| 2024.01 | **Gene-associated Disease Discovery Powered by Large Language Models** |   arXiv    | [link](https://arxiv.org/abs/2401.09490) |                              -                               |



####  Identification of Transcription Factor Binding Sites
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.08 | **BertSNR: an interpretable deep learning framework for single-nucleotide resolution identification of transcription factor binding sites based on DNA language model** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae461) |                              [link](https://github.com/lhy0322/BertSNR)                               |
| 2024.05 | **BERT-TFBS: a novel BERT-based model for predicting transcription factor binding sites by transfer learning** |   Briefings in Bioinformatics    | [link](https://doi.org/10.1093/bib/bbae195) |                              [link](https://github.com/ZX1998-12/BERT-TFBS)                               |
| 2024.01 | **Multiomics-integrated deep language model enables in silico genome-wide detection of transcription factor binding site in unexplored biosamples** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae013) |                              -                               |



#### Origins of Replication Rite Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.01 | **PLANNER: a multi-scale deep language model for the origins of replication site prediction** |   IEEE Journal of Biomedical and Health Informatics    | [link](https://ieeexplore.ieee.org/abstract/document/10380693) |                        -                               |

#### DNA-binding Protein Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **Improving prediction performance of general protein language model by domain-adaptive pretraining on DNA-binding protein** |   Nature Communications    | [link](https://www.nature.com/articles/s41467-024-52293-7) |                              [link](https://github.com/pengsl-lab/ESM-DBP)                               |
| 2024.07 | **Prediction of Protein-DNA Binding Sites Based on Protein Language Model and Deep Learning** |   International Conference on Intelligent Computing    | [link](https://link.springer.com/chapter/10.1007/978-981-97-5692-6_28) |                              -                              |
| 2024.03 ✨✨✨ | **EquiPNAS: improved protein–nucleic acid binding site prediction using protein-language-model-informed equivariant deep graph neural networks** |   Nucleic Acids Research    | [link](https://doi.org/10.1093/nar/gkae039) |                              [link](https://github.com/Bhattacharya-Lab/EquiPNAS)                             |
| 2024.01 | **Predictive Recognition of DNA-binding Proteins Based on Pre-trained Language Model BERT** |   Journal of Bioinformatics and Computational Biology   | [link](https://doi.org/10.1142/s0219720023500282 ) |                              -                               |
| 2024.01 | **Protein–DNA binding sites prediction based on pre-trained protein language model and contrastive learning** |   Briefings in Bioinformatics   | [link](https://doi.org/10.1093/bib/bbad488) |                              [link](https://github.com/YAndrewL/clape)                               |
| 2022.09 | **Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-training** |   Interdisciplinary Sciences: Computational Life Sciences    | [link](https://link.springer.com/article/10.1007/s12539-022-00537-9) |                              [link](https://github.com/lhy0322/TFBert)                               |



#### RNA Prediction 

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.07 ✨✨✨| **Single-sequence protein-RNA complex structure prediction by geometric attention-enabled pairing of biological language models** |   bioRxiv    | [link](https://doi.org/10.1101/2024.07.27.605468) |                              [link](https://github.com/Bhattacharya-Lab/ProRNA3D-single)                               |
| 2024.05 | **RNAErnie: Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning** |   Nature Machine Intelligence    | [link](https://www.nature.com/articles/s42256-024-00836-4) |                              [link](https://github.com/CatIIIIIIII/RNAErnie)                               |
| 2024.02 | **RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks** |   arXiv    | [link](https://arxiv.org/abs/2403.00043) |                              [link](https://github.com/lbcb-sci/RiNALMo)                               |
| 2023.10 | **Multiple sequence alignment-based RNA language model and its application to structural inference** |   Nucleic Acids Research    | [link](https://academic.oup.com/nar/article/52/1/e3/7369930?login=false) |                             [link](https://github.com/yikunpku/RNA-MSM)                               |
| 2023.07 | **Uni-RNA: Universal Pre-trained Models Revolutionize RNA Research** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.07.11.548588v1) |                              -                               |
| 2023.06 | **Prediction of Multiple Types of RNA Modifications via Biological Language Model** |   TCBB    | [link](https://ieeexplore.ieee.org/document/10146457) |                              [link](https://github.com/abhhba999/MRM-BERT)                               |
| 2023.02 | **Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.31.526427v1) |                             [link](https://github.com/chenkenbio/SpliceBERT)                               |


#### **Sequence Modeling**
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **Toward Understanding BERT-Like Pre-Training for DNA Foundation Models** |  	arXiv    | [link](https://doi.org/10.48550/arXiv.2310.07644)  |        -         |
| 2024.08 | **LitGene: a transformer-based model that uses contrastive learning to integrate textual information into gene representations** |  bioRxiv    | [link](https://doi.org/10.1101/2024.08.07.606674)  |                 [link](https://github.com/vinash85/LitGene)|
| 2024.08 | **BiRNA-BERT allows efficient RNA language modeling with adaptive tokenization** |  bioRxiv    | [link](https://doi.org/10.1101/2024.07.02.601703)  |                 [link](https://github.com/buetnlpbio/BiRNA-BERT)|
| 2024.07 ✨✨✨| **VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling** |  ICML'24    | [link](https://arxiv.org/pdf/2405.10812)  |                              [link](https://github.com/Lupin1998/VQDNA)                           |
| 2024.06 | **Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling** |   ICML’24    | [link](https://arxiv.org/abs/2403.03234) |                              [link](https://github.com/kuleshov-group/caduceus)                               |
| 2024.06 | **Contrastive pre-training for sequence based genomics models** |   bioRxiv    | [link](https://doi.org/10.1101/2024.06.10.598319) |                              [link](https://github.com/ksenia007/cGen)                               |
| 2024.05 | **Dirichlet Flow Matching with Applications to DNA Sequence Design** |   ICML’24    | [link](https://arxiv.org/abs/2402.05841) |                              [link](https://github.com/HannesStark/dirichlet-flow-matching)                               |
| 2024.05 🏋️🏋️| **Self-Distillation Improves DNA Sequence Inference** |   arXiv    | [link](https://arxiv.org/abs/2405.08538) |                              [link](https://github.com/wiedersehne/FinDNA)                               |
| 2024.05 🏋️🏋️| **Accurate and efficient protein embedding using multi-teacher distillation learning** |   arXiv    | [link](https://arxiv.org/abs/2405.11735) |                              [link](https://github.com/KennthShang/MTDP)                               |
| 2024.04 | **Effect of tokenization on transformers for biological sequences** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae196) |                              [link](https://github.com/technion-cs-nlp/BiologicalTokenizers)                               |
| 2024.04 | **A Sparse and Wide Neural Network Model for DNA Sequences** |   SRNN    | [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4806928) |                              [link](https://github.com/wiedersehne/SwanDNA)                               |
| 2024.03 | **Self-supervised learning for DNA sequences with circular dilated convolutional networks** |   Neural Networks    | [link](https://doi.org/10.1016/j.neunet.2023.12.002) |                              [link](https://github.com/wiedersehne/cdilDNA)                               |
| 2024.01 | **ProtHyena: A fast and efficient foundation protein language model at single amino acid Resolution** |   bioRxiv    | [link](https://doi.org/10.1101/2024.01.18.576206) |                              [link](https://github.com/ZHymLumine/ProtHyena)                               |


## Basics of Sequence Modeling
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **dnaGrinder: a lightweight and high-capacity genomic foundation model** |    arXiv   | [link](https://doi.org/10.48550/arXiv.2409.15697)  |                              -                               |
| 2024.02 | **Transformer-VQ: Linear-Time Transformers via Vector Quantization** |   ICLR’24    | [link](https://doi.org/10.48550/arXiv.2309.16354) |                              [link](https://github.com/transformer-vq/transformer_vq)                               |
| 2024.01 | **Scavenging Hyena: Distilling Transformers into Long Convolution Models** |    arXiv  | [link](https://doi.org/10.48550/arXiv.2401.17574) | - |
| 2023.06 | **HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution** |   NeurIPS’23    | [link](https://proceedings.neurips.cc/paper_files/paper/2023/file/86ab6927ee4ae9bde4247793c46797c7-Paper-Conference.pdf) |                              [link](https://github.com/HazyResearch/hyenadna)                               |


## Quantization
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **Low-Rank Quantization-Aware Training for LLMs** |    arXiv  | [link](https://arxiv.org/abs/2406.06385) | [link](https://github.com/Qualcomm-AI-research/lr-qat) |


## Fine-tuning
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.07 | **LoRA+: Efficient Low Rank Adaptation of Large Models** |    ICML'24   | [link](https://icml.cc/virtual/2024/poster/34209) | [link](https://github.com/nikhil-ghosh-berkeley/loraplus) |
| 2021.10 | **LoRA: Low-Rank Adaptation of Large Language Models** |    arXiv   | [link](https://arxiv.org/abs/2106.09685) | [link](https://github.com/microsoft/LoRA) |
| 2024.07 | **DoRA: Weight-Decomposed Low-Rank Adaptation** |   ICML'24    | [link](https://arxiv.org/abs/2402.09353)  |                       [link](https://github.com/NVlabs/DoRA)                       |
| 2024.07 | **Accurate LoRA-Finetuning Quantization of LLMs via Information Retention** |   ICML'24    | [link](https://arxiv.org/abs/2402.05445)  |                     [link](https://github.com/htqin/ir-qlora)                         |
| 2024.05🏋️🏋️| **Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning** |   ACL'24    | [link](https://arxiv.org/abs/2402.13669)  |                     [link](https://github.com/sail-sg/sdft)                         |



## Reducing Knowledge Hallucination
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models** |    ICML'24  | [link](https://openreview.net/forum?id=JCG0KTPVYy) | [link](https://github.com/shiliu-egg/ICML2024_COFT) |



## Other Related Awesome Repository
<ol style="list-style-type: decimal-leading-zero;">
  <li><a href="https://github.com/kebijuelun/Awesome-LLM-Learning">Awesome-LLM-Learning</a></li>
  <li><a href="https://github.com/HICAI-ZJU/Scientific-LLM-Survey">Scientific-LLM-Survey (Biological & Chemical Domains)</a></li>
  <li><a href="https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models">LLM-FineTuning-Large-Language-Models</a></li>
  <li><a href="https://github.com/Curated-Awesome-Lists/awesome-llms-fine-tuning">Awesome-llms-fine-tuning (Explore a comprehensive collection of resources, tutorials, papers, tools, and best practices for fine-tuning Large Language Models (LLMs))</a></li>
  <li><a href="https://github.com/nancheng58/Awesome-LLM4RS-Papers">Awesome-LLM4RS-Papers</a></li>
  <li><a href="https://github.com/WLiK/LLM4Rec-Awesome-Papers">LLM4Rec-Awesome-Papers (A list of awesome papers and resources of recommender system on large language model (LLM))</a></li>
  <li><a href="https://github.com/codefuse-ai/Awesome-Code-LLM">Awesome-Code-LLM (A curated list of language modeling researches for code and related datasets)</a></li>
</ol>





## Contributors

<a href="https://github.com/ychuest" target="_blank"><img src="https://avatars.githubusercontent.com/u/87766116?v=4" alt="ychuest" width="96" height="96"/></a> 

<p align="right">(<a href="#top">back to top</a>)</p>






















