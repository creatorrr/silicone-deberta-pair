---
license: mit
---

# diwank/silicone-deberta-pair

`deberta-large`-based dialog acts classifier. Trained on [silicone-merged](https://huggingface.co/datasets/diwank/silicone-merged): a simplified dialog act datasets from the silicone collection. 

Takes two sentences as inputs (one previous and one current utterance of a dialog). The previous sentence can be an empty string if this is the first utterance of a speaker in a dialog. **Outputs one of 11 labels**:

```python
[
    (0, 'acknowledge')
    (1, 'answer')
    (2, 'backchannel')
    (3, 'reply_yes')
    (4, 'exclaim')
    (5, 'say')
    (6, 'reply_no')
    (7, 'hold')
    (8, 'ask')
    (9, 'intent')
    (10, 'ask_yes_no')
]
```

## Example:

```python
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)

model = ClassificationModel("deberta", "diwank/silicone-deberta-pair")
convert_to_label = lambda n: [
    ['acknowledge',
     'answer',
     'backchannel',
     'reply_yes',
     'exclaim',
     'say',
     'reply_no',
     'hold',
     'ask',
     'intent',
     'ask_yes_no'
    ][i] for i in n
]

predictions, raw_outputs = model.predict([["Say what is the meaning of life?", "I dont know"]])
convert_to_label(predictions)  # answer

```


## Report from W&B

https://wandb.ai/diwank/da-silicone-combined/reports/silicone-deberta-pair--VmlldzoxNTczNjE5?accessToken=yj1jz4c365z0y5b3olgzye7qgsl7qv9lxvqhmfhtb6300hql6veqa5xiq1skn8ys
