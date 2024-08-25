[python-img]: https://img.shields.io/github/languages/top/ychuest/Awesome-LLMs-meet-genomes?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/ychuest/Awesome-LLMs-meet-genomes?color=yellow
[stars-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes/stargazers
[fork-img]: https://img.shields.io/github/forks/ychuest/Awesome-LLMs-meet-genomes?color=lightblue&label=fork
[fork-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=ychuest/Awesome-LLMs-meet-genomes
[adgc-url]: https://github.com/ychuest/Awesome-LLMs-meet-genomes


# Awesome-LLMs-meet-genomes

Awesome-LLMs-meet-genomes is a collection of state-of-the-art, novel, exciting LLMs methods on genomes.  It contains papers, codes, datasets, evaluations, and analyses. Any additional information about LLMs for bioinformatics is welcome, and we are glad to add you to the contributor list [here](#contributors). Any problems, please contact yangchengyjs@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles:

-----------------------------------
[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

## 🔔 News

- 💥 [2024/08] Some real-world experience in training LLMs [link](https://mp.weixin.qq.com/s/ItpCTCcMjTWQJtgpvdwTfw).
- 💥 [2024/08] Three ways of Fine-tuning [link](https://mp.weixin.qq.com/s/MWRW6zZKbK1xJRNsNDOquA).
- 💥 [2024/08] Visualisation of the Transformer Principle [link](https://mp.weixin.qq.com/s/7RBWwf4bQF31E0BySeSx9g).
- 🌟 [2024/08] The Cultivation Method of Large Language Models: A Path to Success [link](https://github.com/wdndev/llm_interview_note).
- 📖 [2024/08] Large Language Models: From Theory to Practice [link](https://ychuest.github.io/Awesome-LLMs-meet-genomes/docs/大规模语言模型：从理论到实践.pdf).


## Important Survey Papers

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2024.07 | **Genomic Language Models: Opportunities and Challenges** |    arXiv   | [Link](https://arxiv.org/abs/2407.11435) |  - |
| 2024.07 | **Scientific Large Language Models: A Survey on Biological & Chemical Domains** |    arXiv   | [Link](https://arxiv.org/abs/2401.14656) |  [link](https://github.com/HICAI-ZJU/Scientific-LLM-Survey) |
| 2024.01 | **Large language models in bioinformatics: applications and perspectives** |    arXiv    | [Link](https://arxiv.org/abs/2401.04155v1) |  - |
| 2023.11 | **To Transformers and Beyond: Large Language Models for the Genome** |    arXiv    | [Link](https://arxiv.org/abs/2311.07621) |  -   |
| 2023.01 | **Applications of transformer-based language models in bioinformatics: a survey** |    Bioinformatics Advances    | [Link](https://arxiv.org/abs/2311.07621) |  -   |



## Papers


### Genomic Large Language Models (Gene-LLMs)



#### General
| Year | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.07 | **Scorpio : Enhancing Embeddings to Improve Downstream Analysis of DNA sequences** |  bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.07.19.604359v1.abstract)  |                              [link](https://github.com/EESI/Scorpio)                            |
| 2024.07 | **DNA language model GROVER learns sequence context in the human genome (可用于蛋白质-DNA结合预测任务)** |   Nature Machine Intelligence    | [link](https://doi.org/10.1038/s42256-024-00872-0)  |                              [link](https://doi.org/10.5281/zenodo.8373202)   [tutorials](https://doi.org/10.5281/zenodo.8373158)                            |
| 2024.04 | **DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome** |   ICLR'24    | [link](https://openreview.net/pdf?id=oMLQB4EZE1)  |                              [link](https://github.com/MAGICS-LAB/DNABERT_2)                               |
| 2024.04 | **Species-aware DNA language models capture regulatory elements and their evolution** |   Genome Biology    | [link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03221-x)  |            [link](https://github.com/gagneurlab/SpeciesLM) |
| 2024.02 | **GenomicLLM: Exploring Genomic Large Language Models: Bridging the Gap between Natural Language and Gene Sequences** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.26.581496v1)  |                                            -                |
| 2024.02 | **Sequence modeling and design from molecular to genome scale with Evo** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1)  |                                             [link](https://github.com/evo-design/evo)                |
| 2024.01 | **ProkBERT family: genomic language models for microbiome applications** |    Frontiers in Microbiology    | [Link](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1331233/full) |  [link](https://github.com/nbrgppcu/prokbert)   |
| 2023.08 | **DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks** |   bioRxiv    | [link](https://www.bioRxiv.org/content/10.1101/2023.07.11.548628v2)  |                                             [link](https://github.com/TencentAILabHealthcare/DNAGPT)                |
| 2023.08 | **Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision** |    arxiv   | [link](https://arxiv.org/abs/2311.02333)  |                              [link](https://github.itap.purdue.edu/Clan-labs/ENBED)                               |
| 2023.07 | **EpiGePT: a Pretrained Transformer model for epigenomics** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.07.15.549134v2)  |                              [link](https://github.com/ZjGaothu/EpiGePT)                               |
| 2023.06 | **Transfer learning enables predictions in network biology** |   nature   | [link](https://www.nature.com/articles/s41586-023-06139-9)  |                              [link](https://github.com/jkobject/geneformer)                               |
| 2023.06 | **GENA-LM: A Family of Open-Source Foundational DNA Language Models for Long Sequences** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v2.abstract)  | [link](https://github.com/AIRI-Institute/GENA_LM)                               |
| 2023.06 | **HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution** |   NIPS'23    | [link](https://dl.acm.org/doi/10.5555/3666122.3667994)  |                              [link](https://github.com/HazyResearch/hyena-dna)                               |
| 2023.01 | **The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)  |                              [link](https://github.com/instadeepai/nucleotide-transformer)                               |
| 2023.01 | **Species-aware DNA language modeling** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v1)  |            [link](https://github.com/DennisGankin/species-aware-DNA-LM)                                                 |
| 2022.08 | **MoDNA: motif-oriented pre-training for DNA language model** |   BCB'22    | [link](https://dl.acm.org/doi/10.1145/3535508.3545512)  |            [link](https://github.com/uta-smile/MoDNA)                                                |
| 2021.02 | **DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome** |   Bioinformatics    | [link](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)  |                              [link](https://github.com/jerryji1993/DNABERT)                               |



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




#### Variants and Evolution Prediction

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2023.10 | **GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction** |   bioRxiv    | [link](https://www.bioRxiv.org/content/10.1101/2023.10.10.561776v1.abstract) |                              [link](https://github.com/clinfo/GPN-MSA-env)                              |
| 2023.10 | **GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics** |   The International Journal of High Performance Computing Applications    | [link](https://journals.sagepub.com/doi/10.1177/10943420231201154) |                              [link](https://github.com/ramanathanlab/genslm)                               |
| 2023.08 | **DNA language models are powerful zero-shot predictors of non-coding variant effects** |   arXiv    | [link](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1) |                              [link](https://github.com/songlab-cal/gpn)                               |


#### Fine-tuning for Genomes
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.02 | **Efficient and Scalable Fine-Tune of Language Models for Genome Understanding** |    arXiv  | [link](https://arxiv.org/abs/2402.08075) | [link](https://github.com/zhanglab-aim/LINGO) |


#### Interaction Prediction
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.04 | **Genomic language model predicts protein co-regulation and function** |   nature communications     | [link](https://www.nature.com/articles/s41467-024-46947-9)  |                               [link](https://github.com/y-hwang/gLM)                              |
| 2024.01 | **Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-training** |   arXiv    | [link](https://arxiv.org/abs/2401.09490) |                              -                               |
| 2022.09 | **Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-training** |   Interdisciplinary Sciences: Computational Life Sciences    | [link](https://link.springer.com/article/10.1007/s12539-022-00537-9) |                              [link](https://github.com/lhy0322/TFBert)                               |




#### RNA Prediction

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.05 | **RNAErnie: Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning** |   Nature Machine Intelligence    | [link](https://www.nature.com/articles/s42256-024-00836-4) |                              [link](https://github.com/CatIIIIIIII/RNAErnie)                               |
| 2024.02 | **RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks** |   arXiv    | [link](https://arxiv.org/abs/2403.00043) |                              [link](https://github.com/lbcb-sci/RiNALMo)                               |
| 2023.10 | **Multiple sequence alignment-based RNA language model and its application to structural inference** |   Nucleic Acids Research    | [link](https://academic.oup.com/nar/article/52/1/e3/7369930?login=false) |                             [link](https://github.com/yikunpku/RNA-MSM)                               |
| 2023.07 | **Uni-RNA: Universal Pre-trained Models Revolutionize RNA Research** |   bioRxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.07.11.548588v1) |                              -                               |
| 2023.06 | **Prediction of Multiple Types of RNA Modifications via Biological Language Model** |   TCBB    | [link](https://ieeexplore.ieee.org/document/10146457) |                              [link](https://github.com/abhhba999/MRM-BERT)                               |
| 2023.02 | **Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction** |   biorxiv    | [link](https://www.biorxiv.org/content/10.1101/2023.01.31.526427v1) |                             [link](https://github.com/chenkenbio/SpliceBERT)                               |


### Quantization
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **Low-Rank Quantization-Aware Training for LLMs** |    arXiv  | [link](https://arxiv.org/abs/2406.06385) | [link](https://github.com/Qualcomm-AI-research/lr-qat) |


### Fine-tuning
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.07 | **LoRA+: Efficient Low Rank Adaptation of Large Models** |    ICML'24   | [link](https://icml.cc/virtual/2024/poster/34209) | [link](https://github.com/nikhil-ghosh-berkeley/loraplus) |
| 2021.10 | **LoRA: Low-Rank Adaptation of Large Language Models** |    arXiv   | [link](https://arxiv.org/abs/2106.09685) | [link](https://github.com/microsoft/LoRA) |
| 2024.07 | **DoRA: Weight-Decomposed Low-Rank Adaptation** |   ICML'24    | [link](https://arxiv.org/abs/2402.09353)  |                       [link](https://github.com/NVlabs/DoRA)                       |
| 2024.07 | **Accurate LoRA-Finetuning Quantization of LLMs via Information Retention** |   ICML'24    | [link](https://arxiv.org/abs/2402.05445)  |                     [link](https://github.com/htqin/ir-qlora)                         |



### Reducing Knowledge Hallucination
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models** |    ICML'24  | [link](https://openreview.net/forum?id=JCG0KTPVYy) | [link](https://github.com/shiliu-egg/ICML2024_COFT) |







## Contributors

<a href="https://github.com/ychuest" target="_blank"><img src="https://avatars.githubusercontent.com/u/87766116?v=4" alt="ychuest" width="96" height="96"/></a> 

<p align="right">(<a href="#top">back to top</a>)</p>






















