<div align="center">
  <img src="mirage_logo.png" width="500"/>
  <h4> Toward faithful answer attribution with model internals 🌴 </h4>
</div>
<br/>
<div align="center">
  
Authors (_ Equal contribution_): [Jirui Qi*](https://betswish.github.io/) • [Gabriele Sarti*](https://gsarti.com/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)  
</div>


> [!TIP]
> The README file is still building in progress. :)
> 
> This repository provides an easy-to-use MIRAGE framework for analyzing the groundedness of RAG generation to the retrieved documents. To re-produce the paper results, please take a look at [this repo](github).


> **Abstract:** Ensuring the verifiability of model answers is a fundamental challenge for retrieval-augmented generation (RAG) in the question answering (QA) domain. Recently, self-citation prompting was proposed to make large language models (LLMs) generate citations to supporting documents along with their answers. However, self-citing LLMs often struggle to match the required format, refer to non-existent sources, and fail to faithfully reflect LLMs' context usage throughout the generation. In this work, we present MIRAGE --Model Internals-based RAG Explanations -- a plug-and-play approach using model internals for faithful answer attribution in RAG applications. MIRAGE detects context-sensitive answer tokens and pairs them with retrieved documents contributing to their prediction via saliency methods. We evaluate our proposed approach on a multilingual extractive QA dataset, finding high agreement with human answer attribution. On open-ended QA, MIRAGE achieves citation quality and efficiency comparable to self-citation while also allowing for a finer-grained control of attribution parameters. Our qualitative evaluation highlights the faithfulness of MIRAGE's attributions and underscores the promising application of model internals for RAG answer attribution.

If you find the paper helpful and use the content, we kindly suggest you cite through:
```bibtex
@inproceedings{Qi2024ModelIA,
  title={Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation},
  author={Jirui Qi and Gabriele Sarti and Raquel Fern'andez and Arianna Bisazza},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270619780}
}
```


## Environment: 
Python: 3.9.19

Packages: `pip install -r requirements.txt`

## Quick Start
For a quick start, you only need to put your RAG input in the `data_input/` folder and run the following **one** command to get the LLM outputs with answer attribution:

```
python mirage.py --f data_input/example.json --config configs/llama2_standard_prompt.yaml
```

The LLM outputs will be saved in the folder `data_input_with_ans/`, and the model internals obtained by MIRAGE will be saved in `internal_res/`.
The parameters are listed below:
- `f`: Path to the input file.
- `config`: Path to the configuration file, containing the generation parameters and prompts.
- `CTI`: How many standard deviations over average for CTI threshold, default `1`.
- `CCI`: Parameter for CCI filtering. Using Top k strategy if k > 0; otherwise Top (-k)% if k < 0.

## Extra Parameters
If you already have LLM generations in the data file, put it in the `data_input_with_ans` and specify the parameter `f_with_ans`.

```bash
mkdir data_input_with_ans
python mirage.py --f data_input_with_ans --config configs/llama2_standard_prompt.yaml --f_with_ans
```


  


