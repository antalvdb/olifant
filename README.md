# Olifant: Memory-based language modeling

This repository contains instructions and code to install, train and run memory-based LLMs. 

Looking for an LLM that is relatively eco-friendly? *Olifant* models have a number of interesting properties:

* **Learning is scalable**, and incremental (allowing checkpointing). Model performance increases approximately log-linearly with more data; model size, learning time and RAM usage co-scale linearly with more data;
* **Consistently low CO2 emissions during training and inference**. *Olifant* runs on CPUs. Training with *Olifant* is estimated to cause in the order of 1,000 times fewer CO2 emissions than neural LM training. Inference costs are 10-100 times lower;
* **Fully transparent functioning**. *Olifant* offers nearest-neighbor-based explanations for predictions, based on individual examples. Nearest neighbors can be source-tagged for full provenance;
* **Intentional memorization of training data**. Depending on context size settings, *Olifant* models can recite the majority of tokens from their training data faithfully.

*Olifant* is our implementation of memory-based language modeling; it is the Dutch word for elephant. To quote [Wikipedia](https://en.wikipedia.org/wiki/Elephant_cognition), "Most contemporary ethologists view the elephant as one of the world's most intelligent animals. Elephants manifest a wide variety of behaviors, including those associated with grief, learning, mimicry, playing, altruism, tool use, compassion, cooperation, self-awareness, memory, and communication."


## Installation

Download and install via pip:

``% pip install olifant``

Alternatively, for the latest development version, clone this repository and run ``pip install .``.

This will automatically pull in and install the Python bindings to TiMBL, [python3-timbl](https://github.com/proycon/python-timbl) (wheels are available for Python versions 3.10, 3.11, and 3.12 on systems with glibc 2.28 or higher; on macOS, installation only works with Python 3.13 currently):

*Olifant* also relies on the command-line version of [TiMBL](https://github.com/LanguageMachines/timbl/) memory-based classification engine for training. 

Install TiMBL on Debian/Ubuntu systems with

``$ sudo apt install timbl``

On Alpine Linux:

``# apk add timbl``

On macOS with brew, invoke

``% brew install timbl``

**Note:** Windows is not supported

## Usage

### Tokenization

Training *Olifant* assumes that you have a tokenizer and a raw-text training set `textfile`. The tokenizer will have to be the same tokenizer used for testing.
First, the text is tokenized using `gpt2` (a standard LLM tokenizer from Hugging Face; we will need to use the same tokenizer in later steps).
Edit `tok.py` if you want to use a different tokenizer.

``% olifant-tok textfile``

This creates a file `textfile_tok` which then needs to be converted to a fixed-width instance base to make it suitable training data for TiMBL.
The example works with an input buffer of 4 tokens, which in current LLM terms is an extremely small input buffer. 

``% olifant-continuous-windowing textfile_tok 4 > textfile_tok.l4r0``

This creates `textfile_tok.l4r0`, creating 4-token windowed instances with the next fifth token as the label to be classified and all previous tokens as context.
Empty lines in the original tokenized text signify the reset of the context window (padded with "_").

### Training

Training can then be invoked by calling TiMBL. This can take a while and may consume high amounts of RAM.

``% timbl -f textfile_tok.l4r0 -a0 -I textfile_tok.l4r0.ibase``

The end result is `textfile_tok.l4r0.ibase`, an indexed and compressed instance base suitable for TiMBL classification. In LLM terms, this is the model file
that you will need for your favorite LLM inference steps.

The option `-a0` means that the training set is compressed losslessly, with compression rates around 10-30%. 
With `-a1`, a strong lossy compression is applied, yielding higher compression levels around 90-95%, and considerably faster but less accurate inference.

### Fine-tuning

Memory-based language models are natural incremental learners, so any learned model can be complemented by additional fine-tuning from any new training set, creating a new `ibase` model. 
This requires a TiMBL invocation similar to the training command; it now includes a previously generated `ibase` model file as starting point. Assuming you
have tokenized and windowed a new training set `finetune_tok.l4r0`:

``% timbl -a0 +D -i textfile_tok.l4r0.ibase -f finetune_tok.l4r0 -I textfile-finetune_tok.l4r0.ibase``

Choose your own naming conventions to keep track of trained and finetuned `ibase` model files. Any `ibase` file created with the `-a0` or `-a4` algorithm options can be the starting point for further finetuning.
This also offers a way to do stepwise training with segments of training data under limited RAM conditions.

### Inference

Simple GPT-style text completion can be invoked by issuing

``% olifant-timbl-llm --classifier textfile-finetune_tok.l4r0 --tokenizer bert-base-cased --timbl_args '-a4 +D' --verbosity 3``

This call assumes the presence of `textfile-finetune_tok.l4r0.ibase`. The arguments passed to the TiMBL engine are '-a4 +D', 
invoking the so-called TRIBL2 k-NN approximation, a relatively fast approximation of *k*-nearest neighbor classification. See the [TiMBL reference guide](https://github.com/LanguageMachines/timbl/blob/master/docs/Timbl_6.4_Manual.pdf) 
for all possible algorithmic variants (-a), the important k parameter (set to 1 by default), and many more options.

You can also run a Jupyter Notebook version:

``% jupyter notebook timbl-llm.ipynb``

Be sure to adjust the way you load your `.ibase` model file.

### Inference, Hugging Face style

In this Jupyter Notebook you see how Olifant can be run Hugging
Face style:

``% jupyter notebook timbl-llm-hf.ipynb``

An excerpt from this code shows how a `TimblHuggingFaceModel` is initialized:

```
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Initialize the Timbl classifier
    classifier = timbl.TimblClassifier(args.classifier, args.timbl_args)
    classifier.load()

    config = AutoConfig.from_pretrained("antalvdb/mblm-chatbot-instruction-prompts-igtree")
    tokenizer.add_special_tokens({'pad_token': '_'})
    tokenizer.pad_token = "_"

    # Initialize the TimblHuggingFaceModel
    model = TimblHuggingFaceModel(config, classifier, tokenizer)
```

### Credits

TiMBL was created 25 years ago by a team that was once the Induction of Linguistic Knowledge group at 
Tilburg University, the Netherlands; members moved to the Computational Linguistics, Psycholinguistics and Sociolinguistics
group at Antwerp University, Belgium, and the Centre for Language and Speech Technology at Radboud University, Nijmegen, 
the Netherlands. Core developer of TiMBL is Ko van der Sloot. Other contributors were Walter Daelemans, Antal van den Bosch, Jakub Zavrel, Peter Berck,
Maarten van Gompel, and many more people credited fully in the [TiMBL reference guide](https://github.com/LanguageMachines/timbl/blob/master/docs/Timbl_6.4_Manual.pdf).

Olifant is described and evaluated in this paper:

> Van den Bosch, A., Risco Patón, A., Buijse, T., Berck, P, and Van Gompel, M. (2025). [Memory-based language models: An efficient, explainable, and eco-friendly approach to large language modeling](https://arxiv.org/abs/2510.22317). arxiv:2510.22317.

Memory-based language modeling was first described in

> Van den Bosch, A. (2005). [Scalable classification-based word prediction and confusible correction](https://pure.uvt.nl/ws/portalfiles/portal/792903/scalable.pdf). *Traitement Automatique des Langues*, 46:2, pp. 39-63.

*Olifant* is a re-implementation of WOPR, a C++ version of a TiMBL-based word predictor developed by Peter Berck,
funded under the NWO Vici project "Memory Models of Language" (2006-2011) awarded to
Antal van den Bosch. Peter Berck wrote a [PhD thesis](https://repository.ubn.ru.nl/bitstream/handle/2066/168708/168708.pdf?sequence=1) on the topic. 
Later, work on memory-based word prediction was
carried forwards by Wessel Stoop and Maarten van Gompel ([Valkuil](https://valkuil.net), [Colibri Core](https://github.com/proycon/colibri-core)).
See this [interactive publication](https://pudding.cool/2019/04/text-prediction/) on autocompletion and next-word prediction.
