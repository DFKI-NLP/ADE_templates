import argparse
import os
import pickle
import sys

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    TextClassificationPipeline

from checklist_customized import CheckListCustomSuite, TimeEntityCreator
from templates_creator import NegationTemplates, PosSentimentTemplates, \
    TempOrderTemplates


def load_entities(filepath):

    with open(filepath, 'rb') as f:
        entity_list = pickle.load(f)
    
    return entity_list


def pred_and_conf_binary(data):
    """Function that gets a prediction from model and reshapes the output for CheckList.

    Taken from CheckList https://github.com/marcotcr/checklist as pred_and_conf(), comments added.
    Function assumes a HF transformer pipeline is used, that the task is a binary classification,
    and that the labels end on integers that corresponds to integer label "LABEL_1" for label=1.

    Args:
        data (str or list[str]): Sentence or sentences to classifiy. Input to model.

    Returns:
        preds: numpy array of predicted integer labels.
        pp: numpy array of prediction values. 
    """

    raw_preds = pipe(data)
    preds = np.array([ int(p["label"][-1]) for p in raw_preds]) # get labels
    pp = np.array([[p["score"], 1-p["score"]] if int(p["label"][-1]) == 0 else [1-p["score"], p["score"]] for p in raw_preds]) # get pred values
    return preds, pp

def get_list_sample(lst, n):
    
    if n == -1:
        return lst
    elif n >= len(lst):
        return lst
    else:
        sample_idx = np.random.choice(a=np.arange(len(lst)), size=n, replace=False)
        list_sample = [lst[idx] for idx in sample_idx]
        return list_sample
    
def get_template_sample(template_dict, n):
    """Helper function for sampling variations of each basic template. 
    Used to reduce the number of templates while keeping a variety for all types of capabailities except beneficial effects.. 

    Args:
        template_dict (dict): Dictionary with basic templates as keys and variations as values. 
            Output of create_all_templates() except for beneficial effects templates.
        n (int): Number of template variations to sample from each basic template. Set to -1 to return all templates.

    Returns:
        list: List of strings of sampled templates.
    """
    
    if n == -1: # no sampling, use all templates
        templates = []
        for basic, vars in template_dict.items():
            templates += vars
        
        return templates
    
    else:
        template_sample = []
        for basic, vars in template_dict.items():
            if n >= len(vars): # use all templates, if too few variations available
                template_sample += vars
            else:
                sample_idx = np.random.choice(a=np.arange(len(vars)), size=n, replace=False)
                vars_sample = [vars[idx] for idx in sample_idx]
                template_sample += vars_sample
        
        return template_sample

def save_sampled_entities(lst, save_to_filename):
    
    sample_saved = open(str(save_to_filename), "w")
    sample_for_saving=map(lambda x:x+'\n', lst)
    sample_saved.writelines(sample_for_saving)
    sample_saved.close()


if __name__== "__main__":

    parser = argparse.ArgumentParser(description="Testing model on selected capabilities for ADE classification.")

    ### data and model sources:

    parser.add_argument(
        '--ade_source', type=str, default="entity_extraction/extraction_results/extracted_psytar_ades.pkl",
        help='Add path to pickle file containing ADEs.'
    )

    parser.add_argument(
        '--mild_ade_source', type=str, default="entity_extraction/extraction_results/extracted_psytar_ades_mild.pkl",
        help='For Positive Sentiment Tests: Add path to pickle file containing less severe ADEs.'
    )

    parser.add_argument(
        '--drug_source', type=str, default="entity_extraction/extraction_results/extracted_psytar_drugs.pkl",
        help='Add path to pickle file containing drug names.'
    )

    parser.add_argument(
        '--model', type=str, 
        default="../model/saved_models/bioredditbert_finetuned_3tweetsdata_batchsample",
        help='Add path to Huggingface model for sequence classification.'
    )

    ### test types:

    parser.add_argument(
        '--temporal_order',
        help='Use flag if temporal order test should be conducted.',
        action='store_true'
    )

    parser.add_argument(
        '--no_temporal_order_time_entity',
        help='For Temporal Order Tests: Use flag if during temporal order tests no additional temporal order test should be conducted using time entities.',
        action='store_true'
    )

    parser.add_argument(
        '--positive_sentiment',
        help='Use flag if ADE + positive sentiment test should be conducted.',
        action='store_true'
    )

    parser.add_argument(
        '--beneficial_effect',
        help='Use flag if beneficial effect test should be conducted.',
        action='store_true'
    )

    parser.add_argument(
        '--true_beneficial_effect_gold_label',type=int, 
        default=0,
        help='For Beneficial Effects Tests: Gold label (0/1) of true beneficial effect statements, depending on annotation scheme.'
    )

    parser.add_argument(
        '--negation',
        help='Use flag if negation test should be conducted.',
        action='store_true'
    )

    ### settings for size of the tests:

    parser.add_argument(
        '--n_samples_templates', type=int, 
        default=1,
        help='Number of samples to draw for each basic template type. Set to -1 to use all. Beneficial Effects tests will always use all templates.'
    )

    parser.add_argument(
        '--n_samples_ade', type=int, 
        default=15,
        help='Number of samples to draw from all available ADEs. Set to -1 to use all.'
    )

    parser.add_argument(
        '--n_samples_drug', type=int, 
        default=5,
        help='Number of samples to draw from all available drug names. Set to -1 to use all.'
    )

    parser.add_argument(
        '--n_samples_time_entities_single', type=int, 
        default=7,
        help='For Temporal Order Tests: Number of time entities to create for fill-in in template.'
    )

    parser.add_argument(
        '--n_samples_time_entities_relational', type=int, 
        default=7,
        help='For Temporal Order Tests: Number of relational time entities to create for fill-in in template.'
    )

    ### other settings:

    parser.add_argument(
        '--seed', type=int, 
        default=89,
        help='Random seed.'
    )
    
    parser.add_argument(
        '--results_folder', type=str, 
        default="checklist_results_01/",
        help='Folder to store results of checklist tests.'
    )

    parser.add_argument(
        '--debug',
        help='Use flag to use small (2) sample sizes for entities for debugging. Number of templates is cropped to 2.',
        action='store_true'
    )    
    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)

    # check results folder name and check if folder exists, if not try to create folder
    if args.results_folder[-1] == "/":
        results_folder = args.results_folder
    else:
        results_folder = str(args.results_folder + "/")
    if not os.path.exists(results_folder):
        try:
            os.makedirs(results_folder)
        except:
            foldererror = str("Folder {} does not yet exist and could not be created. Create folder first, then try again".format(results_folder))
            sys.exit(foldererror)
    
    # set the number of templates to sample
    n_templates_sample = args.n_samples_templates

    # load drug names lists, sample from drug list, save sample to file
    list_drug = load_entities(args.drug_source)
    if args.debug:
        drug_sample = get_list_sample(list_drug, 2)
    elif args.n_samples_drug > -1:
        drug_sample = get_list_sample(list_drug, int(args.n_samples_drug))
    else:
        drug_sample = list_drug
    save_sampled_entities(drug_sample, save_to_filename=str(results_folder+ "drug_sample.txt"))

    # load ADE list depending on the capability to test, sample from ADE list, and save sample to list
    if args.temporal_order or args.negation or args.mild_ade_source == "None":
        list_ade = load_entities(args.ade_source)
        if args.debug:
            ade_sample = get_list_sample(list_ade, 2)
        elif args.n_samples_ade > -1: 
            ade_sample = get_list_sample(list_ade, int(args.n_samples_ade))
        else:
            ade_sample = list_ade
        save_sampled_entities(ade_sample, save_to_filename=str(results_folder+ "ade_sample.txt"))
    if args.positive_sentiment and args.mild_ade_source != "None":
        list_mild_ade = load_entities(args.mild_ade_source)
        if args.debug:
            mild_ade_sample = get_list_sample(list_mild_ade, 2)
        elif args.n_samples_ade > -1: 
            mild_ade_sample = get_list_sample(list_mild_ade, int(args.n_samples_ade))
        else:
            mild_ade_sample = list_mild_ade
        save_sampled_entities(mild_ade_sample, save_to_filename=str(results_folder+ "mild_ade_sample.txt"))

    # load model and HF prediction pipe
    model_checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    if args.temporal_order:

        ### --- TEMPORAL ORDER TESTS --- ###
        print("---------- Starting with TEMPORAL ORDER TESTS ----------")

        # generate temporal order templates
        template_creator = TempOrderTemplates()
        # 1) standard temporal order templates
        temporder_templates_negative = template_creator.create_all_templates(template_type="negative", use_time_entity=False) 
        temporder_templates_positive = template_creator.create_all_templates(template_type="positive",use_time_entity=False)
        # 2) temporal order templates with time entities
        temporder_templates_negative_time = template_creator.create_all_templates(template_type="negative", use_time_entity=True, use_relational_time_entity=False) 
        temporder_templates_positive_time = template_creator.create_all_templates(template_type="positive",use_time_entity=True, use_relational_time_entity=False)
        # 3) temporal order templates with relational time entities
        temporder_templates_negative_time_rel = template_creator.create_all_templates(template_type="negative", use_time_entity=True, use_relational_time_entity=True) 
        temporder_templates_positive_time_rel = template_creator.create_all_templates(template_type="positive",use_time_entity=True, use_relational_time_entity=True)


        # take a sample of the templates
        temporder_templates_negative_sample = get_template_sample(temporder_templates_negative, n_templates_sample)
        temporder_templates_positive_sample = get_template_sample(temporder_templates_positive, n_templates_sample)

        temporder_templates_negative_time_sample = get_template_sample(temporder_templates_negative_time, n_templates_sample)
        temporder_templates_positive_time_sample = get_template_sample(temporder_templates_positive_time, n_templates_sample)
    
        temporder_templates_negative_time_rel_sample = get_template_sample(temporder_templates_negative_time_rel, n_templates_sample)
        temporder_templates_positive_time_rel_sample = get_template_sample(temporder_templates_positive_time_rel, n_templates_sample) 
        
        if args.debug:
            temporder_templates_negative_sample = temporder_templates_negative_sample[0:2]
            temporder_templates_positive_sample = temporder_templates_positive_sample[0:2]
            temporder_templates_negative_time_sample = temporder_templates_negative_time_sample[0:2]
            temporder_templates_positive_time_sample = temporder_templates_positive_time_sample[0:2]
            temporder_templates_negative_time_rel_sample = temporder_templates_negative_time_rel_sample[0:2]
            temporder_templates_positive_time_rel_sample = temporder_templates_positive_time_rel_sample[0:2]
        
        # create single time entities for time templates
        time_creator = TimeEntityCreator()
        if args.debug:
            time_entities_single = time_creator.create_time_entity(n=2, seed=args.seed)
        else:
            time_entities_single = time_creator.create_time_entity(n=args.n_samples_time_entities_single, seed=args.seed)
        # create relational time entities for relational time templates and create small and large list
        if args.debug:
            time_entities_a_smaller_b = time_creator.create_time_entity_pairs(n_pairs=2, initial_seed=args.seed, relation="a_<_b")
        else:
            time_entities_a_smaller_b = time_creator.create_time_entity_pairs(n_pairs=args.n_samples_time_entities_relational, initial_seed=args.seed, relation="a_<_b")
        time_entity_s = []
        time_entity_l = []
        for times in time_entities_a_smaller_b:
            time_entity_s.append(times[0])
            time_entity_l.append(times[1])

        # create testsuite
        custom_suite = CheckListCustomSuite(model_pipe=pipe)

        # create folder to save temporal order results
        save_to = str(results_folder + "temporal_order/")
        
        # run tests
        print("--- Running test with negative templates ---")
        custom_suite.testsuite(templates=temporder_templates_negative_sample,
                entity_fillins={"ade": ade_sample, "drug": drug_sample},
                true_label=0, folder_saveresults=str(save_to + "standard_tests/"))
        print()
        print("--- Running test with positive templates ---")
        custom_suite.testsuite(templates=temporder_templates_positive_sample,
                entity_fillins={"ade": ade_sample, "drug": drug_sample},
                true_label=1, folder_saveresults=str(save_to + "standard_tests/"))
        print()
        print()
        
        if args.no_temporal_order_time_entity:
            print()
            print("Temporal Order Tests COMPLETE.")
        else:
            print("--- Running test with single time entities and negative templates ---")
            custom_suite.testsuite(templates=temporder_templates_negative_time_sample,
                    entity_fillins={"ade": ade_sample, "drug": drug_sample, "time_entity":time_entities_single},
                    true_label=0, folder_saveresults=str(save_to + "time_entity_tests/"))
            print()
            print("--- Running test with single time entities and positive templates ---")
            custom_suite.testsuite(templates=temporder_templates_positive_time_sample,
                        entity_fillins={"ade": ade_sample, "drug": drug_sample, "time_entity":time_entities_single},
                        true_label=1, folder_saveresults=str(save_to + "time_entity_tests/"))
            print()
            print()
            print("--- Running test with relational time entities and negative templates ---")
            custom_suite.testsuite(templates=temporder_templates_negative_time_rel_sample,
                        entity_fillins={"ade": ade_sample, "drug": drug_sample, "time_entity_s": time_entity_s, "time_entity_l": time_entity_l},
                        true_label=0, folder_saveresults=(save_to + "time_entity_rel_tests/"),
                        use_relational_mask_fillers=True, relational_mask_names=["time_entity_s", "time_entity_l"])
            print()
            print("--- Running test with relational time entities and positive templates ---")
            custom_suite.testsuite(templates=temporder_templates_positive_time_rel_sample,
                        entity_fillins={"ade": ade_sample, "drug": drug_sample, "time_entity_s": time_entity_s, "time_entity_l": time_entity_l},
                        true_label=1, folder_saveresults=(save_to + "time_entity_rel_tests/"),
                        use_relational_mask_fillers=True, relational_mask_names=["time_entity_s", "time_entity_l"])

            # save used time entities to file
            time_entities_single_saved = open(str(save_to + "time_entity_tests/" + "time_entities_single.txt"), "w")
            time_entities_single_for_saving=map(lambda x:x+'\n', time_entities_single)
            time_entities_single_saved.writelines(time_entities_single_for_saving)
            time_entities_single_saved.close()

            time_entities_a_smaller_b_saved = open(str(save_to + "time_entity_rel_tests/" + "time_entities_relational_a_smaller_b.txt"), "w")
            time_entities_a_smaller_b_for_saving=map(lambda x:str(x)+'\n', time_entities_a_smaller_b)
            time_entities_a_smaller_b_saved.writelines(time_entities_a_smaller_b_for_saving)
            time_entities_a_smaller_b_saved.close()

            print()
            print("Temporal Order Tests COMPLETE.")
            print()
            print()

    if args.positive_sentiment:
        
        ### --- ADE + POSITIVE SENTIMENT TESTS --- ###
        print("---------- Starting with ADE + POSITIVE SENTIMENT TESTS ----------")

        # Use mild ADEs if provided, use standards ADEs otherwise      
        if args.mild_ade_source != "None":
            print("Positive Sentiment + ADE tests will run with a set of mild ADEs (mild_ade_sample.txt)")
            pos_sent_ade_list = mild_ade_sample
        else: # use standard ADEs 
            pos_sent_ade_list = ade_sample

        # generate templates
        template_creator = PosSentimentTemplates()
        positive_sent_templates = template_creator.create_all_templates()

        # sample templates
        positive_sent_templates_sample = get_template_sample(positive_sent_templates, n_templates_sample)
        
        if args.debug:
            positive_sent_templates_sample = positive_sent_templates_sample[0:2]

        # create testsuite
        custom_suite = CheckListCustomSuite(model_pipe=pipe)

        # create folder to save temporal order results
        save_to = str(results_folder + "positive_sentiment/")
        
        # run tests
        print("--- Running test with ADE + positive sentiment ---")
        custom_suite.testsuite(templates=positive_sent_templates_sample,
                entity_fillins={"ade": pos_sent_ade_list, "drug": drug_sample},
                true_label=1, folder_saveresults=save_to)
        print()
        print()
        print("ADE + Positive Sentiment Tests COMPLETE.")
        print()
        print()
    
    if args.beneficial_effect:

        ### --- BENEFICIAL EFFECT TESTS --- ###
        print("---------- Starting with BENEFICIAL EFFECT TESTS ----------")

        # generate templates
        template_creator = PosSentimentTemplates()
        b_effects_templates_all_dict = template_creator.create_all_templates(beneficial_effects=True)

        # use all templates for each type of beneficial effect
        b_effect_templates_all_sample = {}
        for template_type, template_dict in b_effects_templates_all_dict.items():
            templates = []
            for b_effect, b_effect_templates in template_dict.items():
                templates += b_effect_templates
            if args.debug:
                templates = templates[0:2]
            b_effect_templates_all_sample[template_type] = templates

        # create testsuite
        custom_suite = CheckListCustomSuite(model_pipe=pipe)

        # create folder to save temporal order results
        save_to = str(results_folder + "beneficial_effect/")

        # run tests
        print("--- Running test with with beneficial effects ---")
        custom_suite.testsuite(templates=b_effect_templates_all_sample["beneficial_effect"],
                entity_fillins={"drug": drug_sample},
                true_label=args.true_beneficial_effect_gold_label,
                folder_saveresults=str(save_to + "beneficial/"))
        print() 

        print("--- Running test with non-beneficial effects ---")
        custom_suite.testsuite(templates=b_effect_templates_all_sample["non_beneficial_effect"],
            entity_fillins={"drug": drug_sample},
            true_label=1, folder_saveresults=str(save_to + "non_beneficial/"))
        print()
        print()
        print("Beneficial Effects Tests COMPLETE.")

    if args.negation:
        
        ### --- NEGATION TESTS --- ###
        print("---------- Starting with NEGATION TESTS ----------")

        # generate positive and negative negation templates
        template_creator = NegationTemplates()
        negation_templates_negative = template_creator.create_all_templates(template_type="negative")
        negation_templates_positive = template_creator.create_all_templates(template_type="positive")
       
        # take a sample of the templates
        negation_templates_negative_sample = get_template_sample(negation_templates_negative, n_templates_sample)
        negation_templates_positive_sample = get_template_sample(negation_templates_positive, n_templates_sample)

        if args.debug:
            negation_templates_negative_sample = negation_templates_negative_sample[0:2]
            negation_templates_positive_sample = negation_templates_positive_sample[0:2]
        
        # create testsuite
        custom_suite = CheckListCustomSuite(model_pipe=pipe)

        # create folder to save temporal order results
        save_to = str(results_folder + "negation/")
        
        # run tests
        print("--- Running test with negative templates ---")
        custom_suite.testsuite(templates=negation_templates_negative_sample,
                entity_fillins={"ade": ade_sample, "drug": drug_sample},
                true_label=0, folder_saveresults=save_to)
        print()
        print("--- Running test with positive templates ---")
        custom_suite.testsuite(templates=negation_templates_positive_sample,
                entity_fillins={"ade": ade_sample, "drug": drug_sample},
                true_label=1, folder_saveresults=save_to)
        print()
        print()
        print("Negation Tests COMPLETE.")

    print("All tests completed.")
        



