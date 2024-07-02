# Adaptation of CheckList for ADE Detection

This project adapts [CheckList](https://aclanthology.org/2020.acl-main.442/), a behavioural testing approach, to the task of Adverse Drug Effect (ADE) detection.

Create environment using python version 3.8.10
```
$ pip install -r requirements.txt
```
Install `checklist` with `pip install checklist`.

## Model Fine-tuning
Set up the config file for fine-tuning by adapting the arguments in `model/setup_finetuner_config.py` and running the file. (Or directly adapt the arguments in `model/finetuner_config_bioredditbert.ini` instead.)

Fine-tune by running
```
$ python finetune.py --configfile finetuner_config_bioredditbert.ini
```

## CheckList Adaptation for ADE Detection
In folder `checklist_work/`:

Run `checklist_tests.py` which uses a customized CheckList test suite (`checklist_customized.py`) that uses part of the original CheckList code. 

Run all tests:
```
$ python checklist_tests.py \
    --temporal_order \
    --positive_sentiment \
    --beneficial_effect \
    --true_beneficial_effect_gold_label 0 \
    --negation
```
The Positive Sentiment test will use a ADE fill-ins from a list of less severe ADEs. Deactivate this behaviour if needed:
```
$ python checklist_tests.py \
    --positive_sentiment \
    --mild_ade_source None
```
Inspect default values for sampling of templates and entities as well as other arguments:
```
$ python checklist_tests.py -h
```

Entities to fill the CheckList templates are extracted from the PsyTAR corpus (`entity_extraction/extract_entities.ipynb`). Results are stored in `extraction_results/` and can be directly used for the checklist tests. 
