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
    - [DNA Sequence Design](#dna-sequence-design)
    - [Sequence-to-Function](#sequence-to-function)
    - [Downstream Tasks](#downstream-tasks)
      - [Gene Pathogenicity Prediction](#gene-pathogenicity-prediction)
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
  - [Tokenization](#tokenization)
  - [Position Code](#position-code)
  - [Adversarial Attack](#adversarial-attack)
  - [Quantization](#quantization)
  - [Fine-tuning](#fine-tuning)
  - [Reducing Knowledge Hallucination](#reducing-knowledge-hallucination)
  - [Data processing](#data-processing)
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
| 2025.05 | **Back to recurrent processing at the crossroad of transformers and state-space models** |    Nature Machine Intelligence  | [Link](https://www.nature.com/articles/s42256-025-01034-6) |  - |
| 2025.03 | **Transformers and genome language models** |    Nature Machine Intelligence  | [Link](https://www.nature.com/articles/s42256-025-01007-9) |  - |
| 2025.03 | **Language modelling techniques for analysing the impact of human genetic variation** |    arXiv   | [Link](https://arxiv.org/abs/2503.10655) |  - |
| 2025.03 | **Biological Sequence with Language Model Prompting: A Survey** |    arXiv   | [Link](https://arxiv.org/abs/2503.04135) |  - |
| 2025.01 | **Large Language Models for Bioinformatics** |    arXiv   | [Link](https://arxiv.org/abs/2501.06271) |  - |
| 2024.09 | **Genomic Language Models: Opportunities and Challenges** |    arXiv   | [Link](https://arxiv.org/abs/2407.11435) |  - |
| 2024.07 | **Scientific Large Language Models: A Survey on Biological & Chemical Domains** |    arXiv   | [Link](https://arxiv.org/abs/2401.14656) |  [link](https://github.com/HICAI-ZJU/Scientific-LLM-Survey) |
| 2024.01 | **Large language models in bioinformatics: applications and perspectives** |    arXiv    | [Link](https://arxiv.org/abs/2401.04155v1) |  - |
| 2023.11 | **To Transformers and Beyond: Large Language Models for the Genome** |    arXiv    | [Link](https://arxiv.org/abs/2311.07621) |  -   |
| 2023.01 | **Applications of transformer-based language models in bioinformatics: a survey** |    Bioinformatics Advances    | [Link](https://arxiv.org/abs/2311.07621) |  -   |





## Genomic Large Language Models (Gene-LLMs)



### Generic Base Models
| Year | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.06 | **Generalized biological foundation model with unified nucleic acid and protein language** |    Nature Machine Intelligence  | [link](https://www.nature.com/articles/s42256-025-01044-4)  |                             [link](https://github.com/LucaOne/LucaOne)                             |
| 2025.06 | **eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis** |    ICML'25  | [link](https://arxiv.org/abs/2506.18940)  |                             [link](https://github.com/zzq1zh/GenAI-Lab)                             |
| 2025.06 | **SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model** |    ICLR'25  | [link](https://arxiv.org/abs/2506.01833)  |                             [link](https://github.com/ZhuJiwei111/SPACE)                             |
| 2025.04 | **Bridging Sequence-Structure Alignment in RNA Foundation Models** |    AAAI'25  | [link](https://ojs.aaai.org/index.php/AAAI/article/view/35500)  |                             [link](https://github.com/yangheng95/OmniGenBench)                             |
| 2025.03 | **HydraRNA: a hybrid architecture based full-length RNA language model** |    bioXiv  | [link](https://www.biorxiv.org/content/10.1101/2025.03.06.641765v1)  |                             [link](https://github.com/GuipengLi/HydraRNA)                             |
| 2025.03 | **Pre-training Genomic Language Model with Variants for Better Modeling Functional Genomics** |    bioXiv  | [link](https://www.biorxiv.org/content/10.1101/2025.02.26.640468v2)  |                             [link](https://github.com/HelloWorldLTY/UKBioLM)                             |
| 2025.03 | **Enhancing DNA Foundation Models to Address Masking Inefficiencies** |    arXiv  | [link](https://arxiv.org/abs/2502.18405)  |                             [link](https://github.com/bioscan-ml/BarcodeMAE)                              |
| 2025.02 | **HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model** |    arXiv  | [link](https://arxiv.org/abs/2502.10807)  |                             -                              |
| 2025.02 | **GENERator: A Long-Context Generative Genomic Foundation Model** |    arXiv  | [link](https://arxiv.org/abs/2502.07272)  |                             [link](https://github.com/GenerTeam/GENERator)                              |
| 2025.02 | **Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning** |    arXiv  | [link](https://arxiv.org/abs/2502.03499)  |                             [link](https://huggingface.co/collections/zehui127/omni-dna-67a2230c352d4fd8f4d1a4bd)                              |
| 2025.01 | **MutBERT: Probabilistic Genome Representation Improves Genomics Foundation Models** |    BioXiv  | [link](https://www.biorxiv.org/content/10.1101/2025.01.23.634452v1.abstract)  |                             [link](https://github.com/ai4nucleome/mutBERT)                              |
| 2025.01 | **GENA-LM: a family of open-source foundational DNA language models for long sequences** |    Nucleic Acids Research  | [link](https://academic.oup.com/nar/article/53/2/gkae1310/7954523)  |                             [link](https://github.com/AIRI-Institute/GENA_LM)                              |
| 2024.12 | **EpiGePT: a pretrained transformer-based language model for context-specific human epigenomics** |    Genome Biology   | [link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03449-7)  |                             [link](https://github.com/ZjGaothu/EpiGePT)                              |
| 2024.11 | **VIRALpre: Genomic Foundation Model Embedding Fused with K-mer Feature for Virus Identification** |    bioRxiv   | [link](https://www.biorxiv.org/content/10.1101/2024.11.12.623150v1.abstract)  |                             -                              |
| 2024.11 | **BEACON: Benchmark for Comprehensive RNA Tasks and Language Models** |    NeurIPS'24   | [link](https://openreview.net/forum?id=q2IeJByeSM#discussion)  |                             [link](https://github.com/terry-r123/RNABenchmark)                              |
| 2024.11 | **DNA Language Models for RNA Analyses** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=TOUrnb1EaG)  |                             -                               |
| 2024.10 | **The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling** |    BioRxiv   | [link](https://www.biorxiv.org/content/10.1101/2024.08.14.607850v2)  |                             [link](https://github.com/TattaBio/gLM2)                               |
| 2024.10 | **Character-level Tokenizations as Powerful Inductive Biases for RNA Foundational Models** |    NeurIPS'24   | [link](https://openreview.net/pdf?id=AQ1umQL7dZ)  |                             [link](https://github.com/qiaoqiaoLF/MxDNA)                               |
| 2024.10 | **Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning** |    NeurIPS'24   | [link](https://arxiv.org/abs/2411.02125)  |                             [link](https://github.com/abdcelikkanat/revisitingkmers)                               |
| 2024.10 | **A long-context language model for deciphering and generating bacteriophage genomes** |    Nature Communications   | [link](https://www.nature.com/articles/s41467-024-53759-4)  |                             [link](https://github.com/lingxusb/megaDNA)                               |
| 2024.10 | **Revisiting Convolution Architecture in the Realm of DNA Foundation Models** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=B07dLVWLyD)  |                              -                               |
| 2024.10 | **Hyperbolic Genome Embeddings** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=NkGDNM8LB0)  |                              -                               |
| 2024.10 | **dnaGrinder: a lightweight and high-capacity genomic foundation model** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=phWflQbLhu)  |                              -                               |
| 2024.10 | **DNABERT-S: Pioneering Species Differentiation with Species-Aware DNA Embeddings** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=9klRFLY2TT)  |                              -                               |
| 2024.10 | **Long-range gene expression prediction with token alignment of large language model** |    arXiv   | [link](https://doi.org/10.48550/arXiv.2410.01858)  |                              -                               |
| 2024.09 | **A Comparison of Tokenization Impact in Attention Based and State Space Genomic Language Models** |    bioRxiv   | [link](https://doi.org/10.1101/2024.09.09.612081)  |                              -                               |
| 2024.09 | **Designing realistic regulatory DNA with autoregressive language models** |    Genome Research   | [link](https://genome.cshlp.org/content/early/2024/09/24/gr.279142.124.abstract)  |                    -                               |
| 2024.08 | **Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision** |    Bioinformatics Advances   | [link](https://doi.org/10.1093/bioadv/vbae117)  |                              [link](https://github.itap.purdue.edu/Clan-labs/ENBED)                               |
| 2024.08 | **Unlocking Efficiency: Adaptive Masking for Gene Transformer Models** |  ECAI'24    | [link](https://arxiv.org/pdf/2408.07180)  |                              [link](https://github.com/roysoumya/curriculum-GeneMask)                           |
| 2024.07 | **Genomics-FM: Universal Foundation Model for Versatile and Data-Efficient Functional Genomic Analysis** |  bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.07.16.603653v1)  |                              [link](https://github.com/terry-r123/Genomics-FM)                            |
| 2024.07 ✨✨✨| **VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling** |  ICML'24    | [link](https://arxiv.org/pdf/2405.10812)  |                              [link](https://github.com/Lupin1998/VQDNA)                           |
| 2024.07 | **OmniGenome: Aligning RNA Sequences with Secondary Structures in Genomic Foundation Models** |  arXiv    | [link](https://arxiv.org/abs/2407.11242)  |                              [link](https://github.com/yangheng95/OmniGenomeBench)                           |
| 2024.07 | **Scorpio : Enhancing Embeddings to Improve Downstream Analysis of DNA sequences** |  bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.07.19.604359v1.abstract)  |                              [link](https://github.com/EESI/Scorpio)                            |
| 2024.07 | **DNA language model GROVER learns sequence context in the human genome (可用于蛋白质-DNA结合预测任务)** |   Nature Machine Intelligence    | [link](https://doi.org/10.1038/s42256-024-00872-0)  |                              [link](https://doi.org/10.5281/zenodo.8373202)   [tutorials](https://doi.org/10.5281/zenodo.8373158)                            |
| 2024.05 | **Are Genomic Language Models All You Need? Exploring Genomic Language Models on Protein Downstream Tasks** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.05.20.594989v1)  |                              [link](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species)                           |
| 2024.05 | **GeneAgent: Self-verification Language Agent for Gene Set Knowledge Discovery using Domain Databases** |   arXiv    | [link](https://arxiv.org/pdf/2405.16205)  |              -                       |
| 2024.05 | **DeepGene: An Efficient Foundation Model for Genomics based on Pan-genome Graph Transformer** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.04.24.590879v2.abstract)  |              [link](https://github.com/wds-seu/DeepGene)                       |
| 2024.05 | **Self-Distillation Improves DNA Sequence Inference Databases** |   arXiv    | [link](https://arxiv.org/pdf/2405.08538)  |              [link](https://github.com/wiedersehne/FinDNA)                       |
| 2024.04 | **Effect of tokenization on transformers for biological sequences** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae196)  |                              [link](https://github.com/technion-cs-nlp/BiologicalTokenizers)                               |
| 2024.04 | **DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome** |   ICLR'24    | [link](https://openreview.net/pdf?id=oMLQB4EZE1)  |                              [link](https://github.com/MAGICS-LAB/DNABERT_2)                               |
| 2024.02 | **Exploring Genomic Large Language Models: Bridging the Gap between Natural Language and Gene Sequences** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.26.581496v1)  |                                            [link](https://github.com/Huatsing-Lau/GenomicLLM)  [data](https://zenodo.org/records/10695802)               |
| 2024.02 | **Sequence modeling and design from molecular to genome scale with Evo** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1)  |                                             [link](https://github.com/evo-design/evo)                |\
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



### DNA Sequence Design
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL** |   arXiv    | [link](https://arxiv.org/abs/2505.20578)  |                             -                            |
| 2025.03 | **Regulatory dna sequence design with reinforcement learning** |   ICLR'25    | [link](https://arxiv.org/abs/2503.07981)  |                             [link](https://github.com/yangzhao1230/TACO)                            |
| 2024.05 | **Dirichlet Flow Matching with Applications to DNA Sequence Design** |   ICML'25    | [link](https://arxiv.org/abs/2402.05841)  |                             [link](https://github.com/HannesStark/dirichlet-flow-matching)                            |



### Sequence-to-Function
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.06 | **AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model** |   BioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1.abstract)  |                             [link](https://github.com/google-deepmind/alphagenome)                            |


### Downstream Tasks

#### Gene Pathogenicity Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **PathoLM: Identifying pathogenicity from the DNA sequence through the Genome Foundation Model** |   arXiv    | [link](https://arxiv.org/pdf/2406.13133)  |                             -                            |
| 2024.06 | **Gene Pathogenicity Prediction using Genomic Foundation Models** |   AAAI'24 Spring Symposium on Clinical Foundation Models    | [link](https://www.researchgate.net/profile/Boulbaba-Ben-Amor/publication/381319158_Gene_Pathogenicity_Prediction_using_Genomic_Foundation_Models/links/6669151ca54c5f0b946001ac/Gene-Pathogenicity-Prediction-using-Genomic-Foundation-Models.pdf)  |                             -                            |



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
| 2025.04 | **BioToken and BioFM – Biologically-Informed Tokenization Enables Accurate and Efficient Genomic Foundation Models** |  BioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2025.03.27.645711v1.abstract) |                              [link](https://github.com/m42-health/biofm-eval/)                              |
| 2025.02 | **A SNP Foundation Model: Application in Whole-Genome Haplotype Phasing and Genotype Imputation** |  BioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2025.01.29.635579v1) |                              -                              |
| 2025.01 | **A DNA language model based on multispecies alignment predicts the effects of genome-wide variants** |   Nature Biotechnology    | [link](https://www.nature.com/articles/s41587-024-02511-w) |                              [link](https://github.com/clinfo/GPN-MSA-env)                              |
| 2024.11 | **Leveraging genomic deep learning models for non-coding variant effect prediction** |   ArXiv    | [link](https://arxiv.org/abs/2411.11158)  |            -      |
| 2024.04 | **Species-aware DNA language models capture regulatory elements and their evolution** |   Genome Biology    | [link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03221-x)  |            [link](https://github.com/gagneurlab/SpeciesLM) |
| 2023.10 | **GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics** |   The International Journal of High Performance Computing Applications    | [link](https://journals.sagepub.com/doi/10.1177/10943420231201154) |                              [link](https://github.com/ramanathanlab/genslm)                               |
| 2023.08 | **DNA language models are powerful predictors of genome-wide variant effects** |   PNAS    | [link](https://www.pnas.org/doi/10.1073/pnas.2311219120) |                              [link](https://github.com/songlab-cal/gpn)                               |
| 2022.05 | **SNP2Vec: Scalable Self-Supervised Pre-Training for Genome-Wide Association Study** |   ACL'22    | [link](https://aclanthology.org/2022.bionlp-1.14/) |                              [link](https://github.com/hltchkust/snp2vec)                               |


#### Fine-tuning for Genomes and proteins
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **Fine-tuning sequence-to-expression models on personal genome and transcriptome data** |    bioRxiv  | [link](https://doi.org/10.1101/2024.09.23.614632) | [link](https://github.com/ni-lab/finetuning-enformer) |
| 2024.08 | **Enhancing recognition and interpretation of functional phenotypic sequences through fine-tuning pre-trained genomic models** |    Journal of Translational Medicine  | [link](https://link.springer.com/article/10.1186/s12967-024-05567-z) | [link](https://github.com/GeorgeBGM/Genome_Fine-Tuning) |
| 2024.08 | **Fine-tuning protein language models boosts predictions across diverse tasks** |    Nature Communications  | [link](https://www.nature.com/articles/s41467-024-51844-2) | [link](https://github.com/whatdoidohaha/RFA) |
| 2024.02 | **Efficient and Scalable Fine-Tune of Language Models for Genome Understanding** |    arXiv  | [link](https://arxiv.org/abs/2402.08075) | [link](https://github.com/zhanglab-aim/LINGO) |
| 2023.11 | **Parameter-Efficient Fine-Tune on Open Pre-trained Transformers for Genomic Sequence** |    NeurIPS'23 Workshop GenBio | [link](https://openreview.net/forum?id=HVQoom7ip2) | - |
| 2024.01 | **ViraLM: Empowering Virus Discovery through the Genome Foundation Model** |   bioRxiv    | [link](https://doi.org/10.1101/2024.01.30.577935)  |                                             [link](https://github.com/ChengPENG-wolf/ViraLM)                |


#### Interaction Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.08 | **Large-Scale Multi-omic Biosequence Transformers for Modeling Peptide-Nucleotide Interactions** |   arXiv     | [link](https://arxiv.org/abs/2408.16245)  |                               [link](https://github.com/nyuolab/OmniBioTE)                              |
| 2024.04 | **Genomic language model predicts protein co-regulation and function** |   nature communications     | [link](https://www.nature.com/articles/s41467-024-46947-9)  |                               [link](https://github.com/y-hwang/gLM)                              |
| 2024.01 | **Gene-associated Disease Discovery Powered by Large Language Models** |   arXiv    | [link](https://arxiv.org/abs/2401.09490) |                              -                               |



####  Identification of Transcription Factor Binding Sites
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.03 | **deepTFBS: Improving within- and cross-species prediction of transcription factor binding using deep multi-task and transfer learning** |   BioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2025.03.19.644233v1) |                              [link](https://github.com/cma2015/deepTFBS)                               |
| 2024.10 | **DNA breathing integration with deep learning foundational model advances genome-wide binding prediction of human transcription factors** |   Nucleic Acids Research    | [link](https://doi.org/10.1093/nar/gkae783) |                              [link](https://zenodo.org/records/11130474)                               |
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
| 2025.05 | **HAD: Hybrid Architecture Distillation Outperforms Teacher in Genomic Sequence Modeling** |    arXiv  | [link](https://arxiv.org/abs/2505.20836)  |                             -                             |
| 2025.04 | **Leveraging State Space Models in Long Range Genomics** |    ICLR'25 workshop  | [link](https://arxiv.org/abs/2504.06304)  |                             -                             |
| 2025.03 | **Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences** |    arXiv  | [link](https://arxiv.org/abs/2503.16351)  |                             -                             |
| 2025.02 | **Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification** |    arXiv  | [link](https://arxiv.org/html/2502.07299v1)  |                             -                             |
| 2024.11 | **Unveiling Protein-DNA Interdependency: Harnessing Unified Multimodal Sequence Modeling, Understanding and Generation** |  	-    | -  |        [link](https://github.com/ai4protein/ProDMM)        |
| 2024.09 | **Toward Understanding BERT-Like Pre-Training for DNA Foundation Models** |  	arXiv    | [link](https://doi.org/10.48550/arXiv.2310.07644)  |        -         |
| 2024.08 | **LitGene: a transformer-based model that uses contrastive learning to integrate textual information into gene representations** |  bioRxiv    | [link](https://doi.org/10.1101/2024.08.07.606674)  |                 [link](https://github.com/vinash85/LitGene)|
| 2024.08 | **BiRNA-BERT allows efficient RNA language modeling with adaptive tokenization** |  bioRxiv    | [link](https://doi.org/10.1101/2024.07.02.601703)  |                 [link](https://github.com/buetnlpbio/BiRNA-BERT)|
| 2024.07 ✨✨✨| **VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling** |  ICML'24    | [link](https://arxiv.org/pdf/2405.10812)  |                              [link](https://github.com/Lupin1998/VQDNA)                           |
| 2024.06 💥💥💥 | **Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling** |   ICML’24    | [link](https://arxiv.org/abs/2403.03234) |                              [link](https://github.com/kuleshov-group/caduceus)                               |
| 2024.06 | **Contrastive pre-training for sequence based genomics models** |   bioRxiv    | [link](https://doi.org/10.1101/2024.06.10.598319) |                              [link](https://github.com/ksenia007/cGen)                               |
| 2024.05 | **Dirichlet Flow Matching with Applications to DNA Sequence Design** |   ICML’24    | [link](https://arxiv.org/abs/2402.05841) |                              [link](https://github.com/HannesStark/dirichlet-flow-matching)                               |
| 2024.05 🏋️🏋️| **Self-Distillation Improves DNA Sequence Inference** |   arXiv    | [link](https://arxiv.org/abs/2405.08538) |                              [link](https://github.com/wiedersehne/FinDNA)                               |
| 2024.05 🏋️🏋️| **Accurate and efficient protein embedding using multi-teacher distillation learning** |   arXiv    | [link](https://arxiv.org/abs/2405.11735) |                              [link](https://github.com/KennthShang/MTDP)                               |
| 2024.04 | **Effect of tokenization on transformers for biological sequences** |   Bioinformatics    | [link](https://doi.org/10.1093/bioinformatics/btae196) |                              [link](https://github.com/technion-cs-nlp/BiologicalTokenizers)                               |
| 2024.04 | **A Sparse and Wide Neural Network Model for DNA Sequences** |   SRNN    | [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4806928) |                              [link](https://github.com/wiedersehne/SwanDNA)                               |
| 2024.03 | **Self-supervised learning for DNA sequences with circular dilated convolutional networks** |   Neural Networks    | [link](https://doi.org/10.1016/j.neunet.2023.12.002) |                              [link](https://github.com/wiedersehne/cdilDNA)                               |
| 2024.01 | **ProtHyena: A fast and efficient foundation protein language model at single amino acid Resolution** |   bioRxiv    | [link](https://doi.org/10.1101/2024.01.18.576206) |                              [link](https://github.com/ZHymLumine/ProtHyena)                               |
| 2023.06 | **HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution** |   NeurIPS’23    | [link](https://proceedings.neurips.cc/paper_files/paper/2023/file/86ab6927ee4ae9bde4247793c46797c7-Paper-Conference.pdf) |                              [link](https://github.com/HazyResearch/hyena-dna)                               |


## Basics of Sequence Modeling
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.06 | **Comba: Improving Bilinear RNNs with Closed-loop Control** |    arXiv   | [link](https://arxiv.org/abs/2506.02475)  |                              [link](https://github.com/AwesomeSeq/Comba-triton)                               |
| 2025.05 | **Efficient Long-range Language Modeling with Self-supervised Causal Retrieval** |    ICML'25   | [link](https://arxiv.org/abs/2410.01651)  |                              -                               |
| 2025.04 | **Learnable Multi-Scale Wavelet Transformer: A Novel Alternative to Self-Attention** |    arXiv   | [link](https://arxiv.org/abs/2504.08801)  |                              -                               |
| 2025.03 | **Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts** |    ICLR'25   | [link](https://openreview.net/pdf?id=HKIvuZxGbl)  |                              [link](https://github.com/OpenSparseLLMs/Linear-MoE)                               |
| 2025.02 | **MoM: Linear Sequence Modeling with Mixture-of-Memories** |    arXiv   | [link](https://arxiv.org/abs/2502.13685)  |                              [link](https://github.com/OpenSparseLLMs/MoM)                               |
| 2025.02 | **Linear Attention for Efficient Bidirectional Sequence Modeling** |    arXiv   | [link](https://arxiv.org/abs/2502.16249)  |                              [link](https://github.com/LIONS-EPFL/LION)                               |
| 2024.10 | **LongMamba: Enhancing Mamba's Long-Context Capabilities via Training-Free Receptive Field Enlargement** |    ICLR'25 Conference Submission   | [link](https://openreview.net/forum?id=fMbLszVO1H)  |                              -                               |
| 2024.09 | **Reparameterized Multi-Resolution Convolutions for Long Sequence Modelling** |    arXiv   | [link](https://doi.org/10.48550/arXiv.2408.09453)  |                              -                               |
| 2024.08 | **SE(3)-Hyena Operator for Scalable Equivariant Learning** |    arXiv   | [link](https://doi.org/10.48550/arXiv.2407.01049)  |                              -                               |
| 2024.04 | **LongVQ: Long Sequence Modeling with Vector Quantization on Structured Memory** |    IJCAI'24   | [link](https://doi.org/10.48550/arXiv.2407.01049)  |                              -                               |
| 2024.02 | **Transformer-VQ: Linear-Time Transformers via Vector Quantization** |   ICLR’24    | [link](https://doi.org/10.48550/arXiv.2404.11163) |                              -                               |
| 2024.02 | **MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts** |   arXiv    | [link](https://arxiv.org/abs/2401.04081) |                              -                               |
| 2024.01 | **Scavenging Hyena: Distilling Transformers into Long Convolution Models** |    arXiv  | [link](https://doi.org/10.48550/arXiv.2401.17574) | - |


## Tokenization
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **DNAZEN: Enhanced Gene Sequence Representations via Mixed Granularities of Coding Units** |    	Rxiv   | [link](https://arxiv.org/abs/2505.02206)  |                             [link](https://github.com/oomics/dnazen)                               |
| 2025.04 | **Genomic Tokenizer: Toward a biology-driven tokenization in transformer models for DNA sequences** |    	BioRxiv   | [link](https://www.biorxiv.org/content/10.1101/2025.04.02.646836v1)  |                             [link](https://github.com/dermatologist/genomic-tokenizer)                               |
| 2025.01 | **A partition cover approach to tokenization** |    	arXiv   | [link](https://arxiv.org/abs/2501.06246)  |                             [link](https://github.com/PreferredAI/pcatt)                               |
| 2024.11 | **Enhancing Large Language Models through Adaptive Tokenizers** |    	NeurIPS'24   | [link](https://openreview.net/forum?id=3H1wqEdK4z)  |                             -                               |
| 2024.11 | **Theoretical Analysis of Byte-Pair Encoding** |    	arXiv   | [link](https://arxiv.org/abs/2411.08671)  |                             -                               |
| 2024.10 | **Generation with Dynamic Vocabulary** |    EMNLP'24   | [link](https://arxiv.org/abs/2410.08481)  |                             [link](https://github.com/Maniyantingliu/generation_with_dynamic_vocabulary)                               |
| 2024.10 | **Adaptive BPE Tokenization for Enhanced Vocabulary Adaptation in Finetuning Pretrained Language Models** |    EMNLP'24 Findings   | [link](https://arxiv.org/abs/2410.03258)  |                             [link](https://github.com/chatty831Adapt-BPE)                               |
| 2024.10 | **Model Decides How to Tokenize: Adaptive DNA Sequence Tokenization with MxDNA** |    NeurIPS'24   | [link](https://openreview.net/pdf?id=AQ1umQL7dZ)  |                             [link](https://github.com/qiaoqiaoLF/MxDNA)                               |
| 2024.09 | **BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training** |    NeurIPS'24   | [link](https://arxiv.org/abs/2409.04599)  |                              [link](https://github.com/pchizhov/picky_bpe)                               |
| 2024.09 | **A Comparison of Tokenization Impact in Attention Based and State Space Genomic Language Models** |    bioRxiv   | [link](https://doi.org/10.1101/2024.09.09.612081)  |                              -                               |
| 2024.04 | **Scaffold-BPE: Enhancing Byte Pair Encoding for Large Language Models with Simple and Effective Scaffold Token Removal** |    arXiv  | [link](https://arxiv.org/abs/2404.17808)  |                              [link](https://github.com/Aaron-LHR/Scaffold-BPE)                               |
| 2024.04 | **Effect of tokenization on transformers for biological sequences** |    Bioinformatics  | [link](https://doi.org/10.1093/bioinformatics/btae196)  |                              [link](https://github.com/technion-cs-nlp/BiologicalTokenizers)                               |
| 2024.02 | **Tokenization Is More Than Compression** |    arXiv  | [link](https://papers.cool/arxiv/search?highlight=1&query=tokenization&show=675)  |                              -                               |
| 2023.10 | **Toward Understanding BERT-Like Pre-Training for DNA Foundation Models** |    	arXiv  | [link](https://doi.org/10.48550/arXiv.2310.07644)  |                              -                               |

## Position Code
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **PaTH Attention: Position Encoding via Accumulating Householder Transformations** |    	arXiv  | [link](https://arxiv.org/abs/2505.16381)  |                              [link](https://github.com/fla-org/flash-linear-attention)                               |
| 2025.04 | **Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation** |    	arXiv  |  [link](https://arxiv.org/abs/2504.18857) |                              [link](https://github.com/LuLuLuyi/DPE)                               |
| 2025.04 | **Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Encoding** |    	arXiv  | [link](https://arxiv.org/abs/2504.06308)  |                              -                               |
| 2024.03 | **YaRN: Efficient Context Window Extension of Large Language Models** |    	ICLR'24  | [link](https://openreview.net/forum?id=wHBfxhZu1u)  |                              [link](https://github.com/jquesnelle/yarn)                               |
| 2024.01 | **Roformer: Enhanced transformer with rotary position embedding** |    	Neurocomputing  | [link](https://www.sciencedirect.com/science/article/abs/pii/S0925231223011864)  |                              [link](https://github.com/lucidrains/rotary-embedding-torch)                               |



## Adversarial Attack
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **Fast and Low-Cost Genomic Foundation Models via Outlier Removal** |    ICML'25  | [link](https://arxiv.org/abs/2505.00598) | [link](https://github.com/MAGICS-LAB/GERM) |
| 2025.05 | **GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity Guidance** |    	arXiv  | [link](https://arxiv.org/abs/2505.23839) | [link](https://github.com/zaixizhang/GeneBreaker/tree/main) |


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

## Data processing
1、[从 FASTA 文件中加载并查询基因组序列](https://github.com/meuleman/SynthSeqs/blob/main/make_data/source.py).  
2、[DNABERT2 Fine-Tuning for DHS Specificity Prediction](https://github.com/Kelvinmao/DNABERT_fintune).  
3、[Scaling-Laws-of-Genomic](https://github.com/wds-seu/Scaling-Laws-of-Genomic).  
4、[Deafness-mutation-sites](https://github.com/Cqerliu/Deafness-mutation-sites).  
5、[DNABERT-2_CNN_BiLSTM](https://github.com/Cqerliu/DNABERT-2_CNN_BiLSTM).  
6、[1_Train_HG](https://github.com/mitiau/G-DNABERT/blob/master/1_Train_HG.ipynb).  
7、[dbtk-dnabert](https://github.com/DLii-Research/dbtk-dnabert).  
8、[DNABERT2_Tokenizer](https://github.com/ChaozhongLiu/LBM_Series/blob/main/notebooks/DNABERT2_Tokenizer.ipynb).  
9、[处理序列的R脚本](https://github.com/raphaelmourad/Mistral-DNA/tree/main/scriptR).



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






















