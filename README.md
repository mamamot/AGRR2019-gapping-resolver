# AGRR-2019 Gapping Resolver

This is a system submitted to Dialog Evaluation 2019 [gapping resolution track](https://github.com/dialogue-evaluation/AGRR-2019).

## Description
This system uses an [AWD-LSTM encoder](https://github.com/mamamot/Russian-ULMFit) to create sentence and token representations and runs an MLP classifier and a linear decoder to find sentences with gapping and attempt to resolve it, respectively.

## Metrics
| Corpus        | Binary        | Resolution  |
| ------------- |--------------:| -----------:|
| Train         | TBD           | TBD         |
| Dev           | TBD           | TBD         |
| Test          | TBD           | TBD         |

## Usage
To run the experiment:
1. Install the requirements from `requirements.txt`.
1. Fetch folders `artifacts` and `data` from [Google Drive](https://drive.google.com/drive/folders/1L9HNa7JngGg9DnzynXHKznAudQdzEo9k?usp=sharing) and put it in the root folder of the repostiroty
2. Run `python resolver.py input_file.csv output_file.csv`. The input file should follow the AGRR format.
