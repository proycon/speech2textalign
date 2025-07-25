# Speech 2 Text Alignment

This command-line tool takes an audio file containing speech as input, along with a text file of that covers the same speech.
It will then do automatic speech recognition on the audio, find the exact timestamps of each word, and aligns this information
with the provided text reference. The output is the original text annotated with timestamp (begin,end) on a word level, a simple TSV output will be provided along with an HTML visualisation.

## Installation

Clone this repository, create a virtual environment, and then do:

```
$ pip install .
```

Note that the actual model is currently not included yet.

## Usage

```
$ speech2textalign --model-dir /path/to/model/dir --filename /path/to/input.wav --align /path/to/input.txt
```
