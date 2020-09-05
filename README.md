# radical-ramble

Multimodial dialogue emotion recognition using the MELD dataset.

## Models
For now the goal is to re-implement the baselines for the MELD dataset. These
are:

* TextCNN (T)
* bcLSTM (T, T+A)
* DialogueRNN (T, T+A)

Where `T` indicates a model with text as input, and `T+A` a model with both
text and audio (simultaneously). We will not implement video features, as the
baseline did not.

## Data
Data comes from the [MELD Github](https://github.com/declare-lab/MELD/).
For now we are using only the Raw data, without the pre-computed features
that they also offer.

We perform a slight change in the files structure. The CSV file for a given
split (i.e., test, dev or train) is renamed to `metadata.csv`. The folder
that contains the videos is renamed to `videos`. We also extract the audio
from each video in another folder `audios`.

Each split is contained in a folder with its name. Thus, we have this structure:

```
data/
|-- dev/
    |-- metadata.csv
    |-- videos/
    |-- audios/
|-- train/
    |-- metadata.csv
    |-- videos/
    |-- audios/
|-- test/
    |-- metadata.csv
    |-- videos/
    |-- audios/
```

We maintain the same naming scheme for each video/audio file as the original.
That is, `dia$X_utt$Y`, where `$X` is the dialogue ID and `Y` the utterance
ID. Files with a different naming scheme are ignored.

## Running this project
Use [Poetry](https://python-poetry.org/) to install most of the dependencies:

    $ poetry install

However, Poetry doesn't support the `-f` option from pip, which
is required to install Pytorch, so you have to install it directly:

    $ pip install torch===1.6.0 torchvision===0.7.0 -f https://download pytorch.org/whl/torch_stable.html

