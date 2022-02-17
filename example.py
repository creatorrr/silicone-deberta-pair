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
