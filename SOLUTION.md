# Solution - SMILES 2026 Hallucination Detection - NLP-approach

## Reproducibility instructions:
Commands for running the solution:

```bash
pip install -r requirements.txt
python solution.py
```

Additional requirements:

- nltk tokenizer 'punkt_tab'
- nltk pos-tagger 'averaged_perceptron_tagger_eng'

They are downloaded directly in code

NLTK-library is noted in `requirements.txt`

## Final solution description:

Modified components:

- `solution.py` - functions for text preprocessing and important tokens selection
- `aggregation.py` - cosine distance geometric features extraction (without raw values)
- `probe.py` - improvement of classifier's network

Final approach:

The solution is based on NLP-approach, on part-of-speech tags for defining important tokens in a sequence and considering their values from hidden states.

I define important tokens in text-prompt and in model's response separately, depending on the question word in the question, which implies a certain part of speech (for example, the question word "how" most often refers to adverbs, so I consider tokens with this tag as important for answering the question). Part-of-speech tagging is applied using `nltk` library, important pos-tags for each question word are defined by linguistic rules. Text 'real' tokens are mapped to Qwen BPE tokens, and I consider hidden states from all layers, but only for these tokens ids.

Then, for each layer, two vectors are formed by summing the vectors of the text-prompt tokens and the vectors of the response tokens, and I find the cosine distance between them, which becomes a feature. Thus, it was possible to reduce the number of features from **896** to **24**.

As for the classifier, the replacement of ReLU with ELU and the addition of Dropout were empirically determined.

Finally, test AUROC is about **0.72**, test F1 is **0.81** and test Accuracy - **0.71**.

The choice of approach was based on the need to focus on the tokens that are associated (or connected to)hallucinations. Linguistically, it was possible to assume that meaningful tokens are those that answer the original question. In this case, there was a simple method for parts of speech that did not take into account semantics or complex constructions, which is why the metric remained at about the same level, but the number of features was reduced. In the future, it is possible to try more complex tags, or a semantics-based approach for choosing tokens.

## Experiments and failed attempts: 
- Using PCA based on the values of the last layer and the last token: the metric remained the same, but the number of features was reduced to only **32**, so it became clear that it is needed to find a way to identify the important layers or the important tokens.
- Determining significant layers using Logit Lens (based on the idea that hallucinations appear on later layers) and obtaining values for the last token: the metric became lower, so it was important to understand which tokens should be taken into account. In the future, it is possible consider a combination of Logit Lens and a part-of-speech and/or semantics-based approach.
