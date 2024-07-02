import copy
import os
import re
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
from checklist.editor import Editor
from checklist.test_types import DIR, INV, MFT


class CheckListCustomSuite():
    """A customized CheckList test suite.
    """

    def __init__(self, model_pipe) -> None:
        """Init.

        Args:
            model_pipe (_type_): A function that given a string input returns model predictions in shape [{'label': 'LABEL_1', 'score': 0.9622157216072083}]. For example a Huggingafce transformers pipeline. 
        """

        self.model = model_pipe


    def testsuite(self, templates:List, entity_fillins:Dict, true_label:int, editor=Editor(),
                    use_relational_mask_fillers=False, relational_mask_names=None,
                    checklist_testtype=MFT, checklist_testname="", checklist_capability="", checklist_description="",
                    save_results=True, folder_saveresults="checklist_results/", savesenttofile=False,
                    round_preds=True, round_decimals=4,
                    save_to_unique_filename=False, id_random_seed=27):
        """Fills in templates and runs test using CheckList and given model.

        Args:
            templates (List): A list of templates with placeholders used by CheckList. Ex: "This is a {adj} template."
            entity_fillins (Dict): For every placeholder in the templates, fill-ins are the values in a list. Ex: {"adj": ["great", "boring"]}
            true_label (int): True label for all the template and fill-in combinations. 0 or 1.
            editor (_type_, optional): CheckList Editor object. Defaults to Editor().
            use_relational_mask_fillers (bool, optional): Set to True if a special condition should be applied when filling in template masks. Defaults to False. 
            relational_mask_names (List, optional): If use_relational_mask_fillers is set to True, add mask names that condition should be applied to here. 
                The special condition is applied by only letting the i-th item of every mask filler named in relational_mask_names be applied as template fillers together.
            checklist_testtype (_type_, optional): Either MFT, DIR, INV. Details in CheckList documentation. Defaults to MFT.
            checklist_testname (str, optional): Name of test CheckList. Defaults to "".
            checklist_capability (str, optional): Name of CheckList capability. Defaults to "".
            checklist_description (str, optional): Description for CheckList test. Defaults to "".
            save_results (bool, optional): If True, results of the test will be saved to csv in folder_saveresults. Defaults to True.
            folder_saveresults (str, optional): Folder to save test results, templates and sentence files to. Defaults to "checklist_results/".
            savesenttofile (bool, optional): If True, a separate text file with sentences will be created. Defaults to False.
            round_preds (bool, optional): If True, model predictions are rounded. Defaults to True.
            round_decimals (int, optional): Number of decimals model predictuons are rounded to if round_preds is True. Defaults to 4.
            save_to_unique_filename (bool, optional): Whether to use unique filenames for every saved csv. Defaults to False.
            id_random_seed (int, optional): If save_to_unique_filename is True, use seed to create unique filename.
        
        """
        if not os.path.exists(folder_saveresults):
            try:
                os.makedirs(folder_saveresults)
            except:
                foldererror = str("Folder {} does not yet exist and could not be created. Create folder first, then try again".format(folder_saveresults))
                print(foldererror)
                return None

        if not str(folder_saveresults).endswith("/"):
            folder_saveresults = str(folder_saveresults + "/")
    
        assert checklist_testtype in [MFT, DIR, INV], "checklist_testtype must be one of [MFT, DIR, INV]."

        if checklist_testtype in [DIR, INV]:
            print("Function has not been tested on INV and DIR test types yet. MFT was tested.")

        templates_dict = self._create_template_dict(template_list=templates)
        
        results_df = pd.DataFrame(columns=["template", "testcases", "pred_label", "probs_0","probs_1"])

        if use_relational_mask_fillers: # there is a special relationship between mask_fillers, templates need to be created differently
            
            assert relational_mask_names is not None,  "If arg use_relational_mask_fillers is used, arg relational_mask_names must be provided."
            assert type(relational_mask_names) == list, "If arg use_relational_mask_fillers is used, arg relational_mask_names must be provided as a list."
            for mask_name in relational_mask_names:
                assert mask_name in entity_fillins.keys(), "arg relational_mask_names must be list of keys in entity_fillins"
            
            results_df, reports, passed_tests_total = self._relational_mask_filler_loop(folder_saveresults=folder_saveresults, templates_dict=templates_dict,
                                                                    entity_fillins=entity_fillins, results_df=results_df,true_label=true_label,
                                                                    relational_mask_names=relational_mask_names, editor=Editor(),
                                                                    checklist_testtype=checklist_testtype, checklist_testname=checklist_testname,
                                                                    checklist_capability=checklist_capability,checklist_description="checklist_description",
                                                                    savesenttofile=savesenttofile, round_preds=round_preds, round_decimals=round_decimals)
        
        else: # create templates in the standard way of combining all mask fillers without special condition
            reports = []
            passed_tests_total = {"passed": 0, "total":0}
            for k, template in templates_dict.items():   
                # fill template
                test_editor = editor.template(templates=template, **entity_fillins,
                                labels=true_label, meta=True, save=True)
            
                # create testsuite
                testsuite = checklist_testtype(**test_editor, name=str(str(k)+str(checklist_testname)), capability = str(checklist_capability),
                    description=str(checklist_description))
            
                if savesenttofile:
                    # save the test sentences to a file
                    filename_sentences = str(str(folder_saveresults)+"sentences_"+ str(k)+".txt")
                    testsuite.to_raw_file(path=filename_sentences)
                else:
                    filename_sentences = None

                # save the test sentences as list
                testcases = self._flatten_list(testsuite.data)

                if savesenttofile:
                    # get model predictions
                    docs = open(filename_sentences).read().splitlines()
                    preds, probs = self._get_preds_probs_arrays(docs, round=round_preds, decimals=round_decimals)
                else:
                    preds, probs = self._get_preds_probs_arrays(testcases, round=round_preds, decimals=round_decimals)
                
                # save sentences and predictions to df
                results_df_new = pd.DataFrame(data={"template": str(k), "testcases": testcases, "pred_label": preds, "probs_0": probs[:, 0],
                                                "probs_1": probs[:, 1]})
                results_df = pd.concat([results_df, results_df_new])

                # print test report, update test report and number of passed test cases
                new_report, pass_and_total = self._create_test_report(preds, true_label=true_label, test_name=k)
                reports.append(pass_and_total)
                passed_tests_total["passed"] = passed_tests_total["passed"] + pass_and_total[0]
                passed_tests_total["total"] = passed_tests_total["total"] + pass_and_total[1]

        print()
        print("Passed {} out of {} testcases.".format(passed_tests_total["passed"], passed_tests_total["total"]))
        print()
        
        if save_results:
            # save the dataframes of results
            if save_to_unique_filename:  
                today = str(date.today()).replace("-", "_")
                np.random.seed(id_random_seed)
                random_id = str(np.random.randint(low=0, high=100000))
                id_results = str(today + "_id" + random_id)

            results_df.reset_index(drop=True, inplace=True)
            
            if save_to_unique_filename:
                path_to_results_df = str(folder_saveresults + "results_df_label"+ str(true_label)+ "_" + id_results + ".csv")
            else:
                path_to_results_df = str(folder_saveresults + "results_df_label"+ str(true_label)+ ".csv")
            
            results_df.to_csv(path_or_buf=path_to_results_df)

            # save the templates in a df together with the test reports
            reports_sorted = sorted(reports, key= lambda items: items[2]) # sort list by template_id           
            reports_template_id_col = []
            reports_templates_col = []
            reports_total_col = []
            reports_passed_col = []

            for testcasereport in reports_sorted:
                reports_template_id_col.append(testcasereport[2])
                reports_templates_col.append(templates_dict[testcasereport[2]])
                reports_total_col.append(testcasereport[1])
                reports_passed_col.append(testcasereport[0])

            templates_and_reports_df = pd.DataFrame(data= {"template_id": reports_template_id_col, "template": reports_templates_col, "total_tests": reports_total_col, "passed_tests": reports_passed_col})
            
            if save_to_unique_filename:
                path_to_templates_and_report_df = str(folder_saveresults + "reports_templates_label" + str(true_label) + "_" + id_results + ".csv")
            else:
                path_to_templates_and_report_df = str(folder_saveresults + "reports_templates_label" + str(true_label) + ".csv")

            templates_and_reports_df.to_csv(path_or_buf=path_to_templates_and_report_df)
            #templates_df = pd.DataFrame(data = {"template_id":templates_dict.keys(), "template": templates_dict.values()})
            #path_to_templates_df = str(folder_saveresults + "templates_label" + str(true_label) + "_" + id_results + ".csv")
            #templates_df.to_csv(path_or_buf=path_to_templates_df)
            
            # save the test reports
            #reports_template_col = [r1[2] for r1 in passed_tests_total]
            #reports_total_col = [r2[1] for r2 in passed_tests_total]
            #reports_passed_col = [r3[0] for r3 in passed_tests_total]
            #reports_df = pd.DataFrame(data = {"template_id": reports_template_col, "total_tests": reports_total_col, "passed_tests": reports_passed_col})
            #path_to_reports = str(str(folder_saveresults + "reports_label" + str(true_label) + "_" + id_results + ".csv"))
            #reports_df.to_csv(path_or_buf=path_to_reports)
            #if os.path.exists(path_to_reports):
            #    print("Warning: Reports file already exists, reports will be appended for file {}, other files will be replaced.".format(path_to_reports))
            #report_header = str("-----Report of results for {}------".format(id_results))
            #reports.insert(0, report_header)
            #for report in reports:
            #    with open(path_to_reports, "a") as f:
            #        f.write(str("\n"+report))

            print()
            print("Results saved in:")
            print(path_to_results_df)
            print(path_to_templates_and_report_df)
    
    def _relational_mask_filler_loop(self, folder_saveresults, results_df, templates_dict, entity_fillins, true_label, relational_mask_names,
                                               editor, checklist_testtype, checklist_testname, checklist_capability,
                                               checklist_description, savesenttofile, round_preds, round_decimals):
        
        # make sure all special condition mask have the same number of entities
        n_entity_tuples = len(entity_fillins[relational_mask_names[0]])
        for name_of_mask in relational_mask_names:
            assert len(entity_fillins[name_of_mask]) == n_entity_tuples, "All lists in entity_fillins that correspond to relational_mask_names must have the same length."

        reports = []
        passed_tests_total = {"passed": 0, "total":0}

        for entity_tuple_idx in range(n_entity_tuples): # loop over each relational entity pair (tuple)

            # create new entity fillin dict for template creation that only has the current relational entity tuple
            new_entity_fillins = copy.deepcopy(entity_fillins)
            for mask_name in relational_mask_names:
                new_entity_fillins[mask_name] = entity_fillins[mask_name][entity_tuple_idx]
            print("Test results for new combination of relational entities:")
            
            for k, template in templates_dict.items():   
                # fill template
                test_editor = editor.template(templates=template, **new_entity_fillins,
                                labels=true_label, meta=True, save=True)
            
                # create testsuite
                testsuite = checklist_testtype(**test_editor, name=str(str(k)+str(checklist_testname)), capability = str(checklist_capability),
                    description=str(checklist_description))
            
                if savesenttofile:
                    # save the test sentences to a file
                    filename_sentences = str(str(folder_saveresults)+"sentences_"+ str(k)+".txt")
                    testsuite.to_raw_file(path=filename_sentences)
                else:
                    filename_sentences = None

                # save the test sentences as list
                testcases = self._flatten_list(testsuite.data)

                if savesenttofile:
                    # get model predictions
                    docs = open(filename_sentences).read().splitlines()
                    preds, probs = self._get_preds_probs_arrays(docs, round=round_preds, decimals=round_decimals)
                else:
                    preds, probs = self._get_preds_probs_arrays(testcases, round=round_preds, decimals=round_decimals)
                
                # save sentences and predictions to df
                results_df_new = pd.DataFrame(data={"template": str(k), "testcases": testcases, "pred_label": preds, "probs_0": probs[:, 0],
                                                "probs_1": probs[:, 1]})
                
                results_df = pd.concat([results_df, results_df_new])

                # print test report, update test report and number of passed test cases
                new_report, pass_and_total = self._create_test_report(preds, true_label=true_label, test_name=k)
                reports.append(pass_and_total) 
                passed_tests_total["passed"] = passed_tests_total["passed"] + pass_and_total[0]
                passed_tests_total["total"] = passed_tests_total["total"] + pass_and_total[1]
        
        # add up passed and total test case count of templates that were used multiple times with different relational entities
        results_dict = dict()
        for r in reports:
            if r[2] in results_dict.keys(): 
                old = results_dict[r[2]]
                results_dict[r[2]] = [r[0] + old[0] , r[1] +old[1]]
            else:
                results_dict[r[2]] = [r[0], r[1]]
        reports_merged = [(v[0], v[1], k) for k,v in results_dict.items()]
        

        return results_df, reports_merged, passed_tests_total
    
    def _get_preds_probs_arrays(self, input, round=True, decimals=4):
        """
        Gets prediction from _pred_and_conf_binary and returns array of (rounded) predictions."
        """
        
        prob_lst = []
        pred_lst = []
        for x in input:
            preds, pp = self._pred_and_conf_binary(x)
            prob_lst.append(pp[0])
            pred_lst.append(preds[0])
        
        preds_array = np.array(pred_lst)
        probs_array = np.array(prob_lst)
        
        if round:
            preds_array = np.round(preds_array, decimals=decimals)
            probs_array = np.round(probs_array, decimals=decimals)
        
        return preds_array, probs_array
        
    def _pred_and_conf_binary(self, data):
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

        raw_preds = self.model(data)
        preds = np.array([ int(p["label"][-1]) for p in raw_preds]) # get labels
        pp = np.array([[p["score"], 1-p["score"]] if int(p["label"][-1]) == 0 else [1-p["score"], p["score"]] for p in raw_preds]) # get pred values
        return preds, pp
    

    def _create_template_dict(self, template_list):
        """_
        Create dictionary of template and enumerate the templates.
        """
        assert type(template_list) in [str, list], "templates input to testsuite() must be either a list of str (multiple templates) or a str (single template)."
        
        if type(template_list) == list:
            template_dict = {}
            for i in range(len(template_list)):
                template_dict[str("template_" + str(i).zfill(3))] = template_list[i]
        else:
            template_dict = {"template_000": template_list}
        
        return template_dict

    def _flatten_list(self, item, data_type=str) -> list:
        """Helper function for create_NP_tagsets() that takes a list and flattens all nested lists of the list if applicable. 

        Args:
            item (list): List that should be flattened (i.e., a list of lists.)
            data_type (_type_, optional): Define data type of the elements on the lowest level of the input item. Defaults to str.

        Returns:
            list: Flattened (not nested) list (a list where all elements are of type data_type).
        """
        
        flattened_list = []

        if type(item) == data_type:
            # base case: item is a str
            flattened_list.append(item)
        else:
            # recursive case
            for element in item:
                flattened_list += self._flatten_list(element, data_type=data_type)
        
        return flattened_list

    def _create_test_report(self, preds, true_label, test_name="Unnamed test"):
        """
        Pints test report for a single template, counting failed testcases. Returns report (str) and tuple of (passed tests, total tests).
        """
        # count correct overall predictions (all testcases)
        if true_label == 0:
            correct_preds = len(preds) - np.count_nonzero(preds)
        else: 
            correct_preds = np.count_nonzero(preds)
        
        report = "{}:  predicted correctly {} out of {} test cases.".format(test_name, correct_preds, len(preds))
        pass_and_total = (correct_preds, len(preds), test_name)
        
        #print(test_name, ":  predicted correctly {} out of {} test cases.".format(correct_preds, len(preds)))
        print(report)

        return report, pass_and_total
  

class TimeEntityCreator():
    """Creates time entities ("3 days", "1 year") or pairs of time entities with a specified relation.
    """

    def __init__(self) -> None:
        pass
    
    def create_time_entity(self, n:int, seed=72) -> List:
        """Creates n time entities.

        Args:
            n (int): Number of time entities to create
            seed (int, optional): Random seed. Defaults to 72.

        Returns:
            List: List of n time entities.
        """

        np.random.seed(seed=seed)
        time_entitiy_names = ["days", "weeks", "months", "years"]
        times = []
        count = 0

        while count < n:
            entity_type = time_entitiy_names[np.random.randint(low=0, high=3)]
            num = np.random.randint(low=1, high=26)
            if num == 1:
                entity_type = entity_type[:-1]
            times.append(str(num) + " " + entity_type)
            count+=1

        return list(times)

    def create_time_entity_pairs(self, n_pairs:int, initial_seed=63, relation="a_<_b") -> List:
        """Creates n_pairs time entity pairs with the specified relation.

        Args:
            n_pairs (int): Number of time entity pairs to create.
            initial_seed (int, optional): Random seed. Defaults to 63.
            relation (str, optional): Expected relation between the elements of the time entity pair. Options: 'a_<_b', 'a_>_b', 'a_==_b'. Defaults to "a_<_b".

        Returns:
            List: List of n_pairs time entity pairs (tuples).
        """

        assert relation in ["a_<_b", "a_>_b", "a_==_b"], "relation argument must be one of ['a_<_b', 'a_>_b', 'a_==_b']"
        check_relation = self._get_relation_func(relation=relation)

        np.random.seed(seed=initial_seed)

        seeds = np.random.randint(low=0, high=10000, size=100*n_pairs)    
        correct_relations = []

        for seed_idx in seeds:
            
            time_entity_1 = self.create_time_entity(n=1, seed=seed_idx)
            time_entity_2 = self.create_time_entity(n=1, seed=seed_idx+1)
            if check_relation(self._convert_to_days(time_entity_1[0]), self._convert_to_days(time_entity_2[0])):
                correct_relations.append((time_entity_1[0], time_entity_2[0]))
            if len(correct_relations) == n_pairs:
                break
        
        return correct_relations

    def _convert_to_days(self, time):
        
        number_pattern = "^[0-9]+"
        number = int(re.findall(number_pattern, time)[0])
        if "day" in time:
            days = number
        elif "week" in time:
            days = number*7
        elif "month" in time:
            days = number*30
        elif "year" in time:
            days = number*365
        
        return days
    
    def _get_relation_func(self, relation="a_<_b"):

        if relation == "a_<_b":
            def relation_func(input_a, input_b):
                if input_a < input_b:
                    return True
                else:
                    return False
        elif relation == "a_>_b":
            def relation_func(input_a, input_b):
                if input_a > input_b:
                    return True
                else:
                    return False
        elif relation == "a_==_b":
            def relation_func(input_a, input_b):
                if input_a == input_b:
                    return True
                else:
                    return False
        else:
            print("relation argument must be one of ['a_<_b', 'a_>_b', 'a_==_b']")
            return None

        return relation_func
