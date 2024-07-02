import configparser

if __name__ == "__main__":

    config = configparser.ConfigParser()

    # on data_columns:
    # save the data columns that will be needed for fine-tuning the model after tokenization
    # this will most likely be the 'label' column and 
    # the  columns (BERT:'input_ids', 'token_type_ids', 'attention_mask') created by the tokenizer

    config["default"] = {"data_raw_format": "csv",
                        "traindata": "../data/new_dataset_3_tweets_splits/new_dataset_trainsplit.csv",
                        "valdata": "../data/new_dataset_3_tweets_splits/new_dataset_valsplit.csv",
                        "testdata": "../data/new_dataset_3_tweets_splits/new_dataset_testsplit.csv",
                        "modelcheckpoint": "cambridgeltl/BioRedditBERT-uncased",
                        "text_columns": "text",
                        "data_columns": "label, input_ids, token_type_ids, attention_mask",
                        "num_labels": "2",
                        "num_train_epochs":"3.0",
                        "train_batch_size": "8",
                        "eval_metric_training": "accuracy",
                        "eval_metric_testset" : "accuracy, f1_binary, f1_weighted",
                        "eval_confusion_matrix" : "True",
                        "eval_confusion_matrix_label_names" : "No ADE (0), ADE (1)",
                        "eval_confusion_matrix_title" : "Testset Results",
                        "output_dir" : "saved_models/bioredditbert_finetuned_3tweetsdata_batchsample",
                        "tokenizer_truncation": "False",
                        "tokenizer_delete_long_inputs": "True", 
                        "max_input_length": "None",
                        "batch_sampling": "True"}



    with open('finetuner_config.ini', 'w') as configfile:
        config.write(configfile)