import numpy as np
import spacy
from itertools import combinations, product
import json
import pickle


class ADE_NP_Extraction():
    def __init__(self, ade_list: list, spacy_model="en_core_web_sm") -> None:
        self.ade_list = list(set(ade_list)) # all unique ADEs

        try:
            self.spacy_pipe = spacy.load(spacy_model)
        except:
            raise Exception("Specified spacy model is not installed. Use: python -m spacy download <spacy_model>")

    def get_all_spacy_pos_tags(self, output_filename="spacy_pos_tags.txt"):
        """Gets all POS tags and descriptions for the given spacy model and saves to file.

        Args:
            output_filename (str, optional): Output file to write POS tags to. Defaults to "spacy_pos_tags.txt".
        """

        file = open(output_filename, 'w')

        pos_tag_list = list(self.spacy_pipe.get_pipe("tagger").labels)
        tag_dict = {}
        for t2 in pos_tag_list:
            tag_dict[t2] = spacy.explain(t2)
        file.write("All fine-grained POS tags: \n")
        for t3, ex in tag_dict.items():
            file.write(str(t3 + " " + ex + "\n"))

        file.close()

    def get_detailed_pos_tags_nouns_adj_adv(self, input_filename="detailed_pos_tags.json"):
        """Read the corresponding spacy POS tags for "noun", "adj", "adv".  

        Args:
            input_filename (str, optional): Json file where detailed POS tags are stored. Defaults to "detailed_pos_tags.json".

        Returns:
            list: A list for of detailed POS tags for noun, adj, adv respectively.
        """

        with open(input_filename, "r") as read_file:
            pos_tags = json.load(read_file)

        return pos_tags["noun_tags_detailed"], pos_tags["adj_tags_detailed"], pos_tags["adv_tags_detailed"]
    
    def get_starter_pos_tags(self, input_filename="starter_pos_tagsets.json"):
        """Read the starter tagsets for nouns, determiners and possessive pronouns, and noun modifiers.  

        Args:
            input_filename (str, optional): Json file where starter tagsets are stored. Defaults to "starter_pos_tagsets.json".

        Returns:
            list: A list od starter tagsets for nouns, determiners and possessive pronouns, and noun modifiers respectively.
        """
    
        with open(input_filename, "r") as read_file:
            tagsets = json.load(read_file)

        return tagsets["nouns"], tagsets["det_and_poss"], tagsets["modifiers"]
    
    def flatten_list(self, item, data_type=str) -> list:
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
                flattened_list += self.flatten_list(element, data_type=data_type)
        
        return flattened_list
    
    def get_noun_idx_list(self, tagset: list[str])-> list[int]:
        """
        Helper function for create_NP_tagsets().
        A function that checks all nouns in a tagset and saves their indices if the noun is a single noun or the first noun before another noun.
        This means the noun_idx lists for in these cases ["noun"], ["adj", "noun"], ["noun", "noun"] are [0], [1], [0] respectively. 
        Second nouns after a noun are excluded because mostly ["noun", "noun"] is a compound noun and should be treated as one entity. 

        tagset: a list of str. Example: ["adj", "noun"]

        returns: Indices as a list of integers. 
        """

        noun_idx = []
        for tag_idx in range(len(tagset)):
            if tagset[tag_idx] == "noun":
                if tag_idx == 0: # noun is single or first nouns because of position
                    noun_idx.append(tag_idx)
                elif tagset[int(tag_idx -1)] != "noun": # noun is single or first noun because previous tag is not a noun
                    noun_idx.append(tag_idx)
                else: # noun is second noun in noun noun comb
                    pass
        
        return noun_idx
    
    def create_noun_compounds(self, nouns: list) -> list:
        """
        Helper function for create_NP_tagsets().
        Creates "noun + noun" compounds of single "noun"

        nouns: List of lists of str. A list that contains tagsets including nouns. Example tagset: ["noun", "IN", "noun"].

        returns: List of lists. List of new modified tagsets. 
        """

        # create combinations of noun and modfiers to add to the list of desired tags
        NP_tags_nouns = nouns[:] # copy list of nouns

        
        for tags in nouns:
            # get all idx of nouns in a tagset
            noun_idx = []
            for tag_idx in range(len(tags)):
                if tags[tag_idx] == "noun":
                    noun_idx.append(tag_idx)
            
            # create new tags with all combinations of noun and noun-noun
            for le in range(1, len(noun_idx)+1):
                for co in combinations(noun_idx, le):
                    new_tag = tags[:]
                    for replace_idx in co:
                        new_tag[replace_idx] = ["noun", "noun"]
                    NP_tags_nouns.append(new_tag)

        return NP_tags_nouns
    
    def create_det_poss_mods(self, NP_tags_nouns_flat: list, det_and_poss: list) -> list:
        """ 
        Helper function for create_NP_tagsets().
        Creates determiner/possessive p.  + noun modifications

        NP_tags_nouns_flat: List of lists of str. Flattened output of create_noun_compounds().
        det_and_poss: List of lists of str. A list that contains tagsets with nouns and determiners/possessive pronouns. Example tagset: ["DT", "noun"].

        returns: List of lists. List of new modified tagsets. 
        """
        NP_tags_det_poss = []
        det_and_poss.append(False) # add a False to list to have "no replacement" as an option

        for tags2 in NP_tags_nouns_flat:

            noun_idx = self.get_noun_idx_list(tags2)
                    
            # create all combinations of det and poss to replace noun
            # for each index create options for replacing
            comb_for_idx_lists = []
            for idx in noun_idx:
                comb_for_idx = [prod for prod in product([idx], det_and_poss)]
                comb_for_idx_lists.append(comb_for_idx)
            
            # create all combinations replacements
            noun_replace_combs = [n_r_c for n_r_c in product(*comb_for_idx_lists)]
            
            for noun_replace_comb in noun_replace_combs:
                new_tags = tags2[:]
                for replacement_tuple in noun_replace_comb:
                    if replacement_tuple[1] == False:
                        pass
                    else: 
                        new_tags[replacement_tuple[0]] = replacement_tuple[1]

                NP_tags_det_poss.append(new_tags)
        
        return NP_tags_det_poss
    
    def create_modifier_mod(self, NP_tags_det_poss_flat: list, modifiers: list) -> list:
        """ 
        Helper function for create_NP_tagsets().
        Creates modifier (+ det/poss.p.) + noun modifications

        NP_tags_det_poss_flat: List of lists of str. Flattened output of create_det_poss_mods().
        modifiers: List of lists of str. A list that contains tagsets with nouns and modifiers. Example tagset: ["adj", "noun"].

        returns: List of lists. List of new modified tagsets. 
        """

        NP_tags_modified = []
        modifiers.append(False)

        for tags3 in NP_tags_det_poss_flat:

            noun_idx = self.get_noun_idx_list(tags3)
        
            # create all combinations of modifiers to replace noun
            # for each index create options for replacing
            comb_for_idx_lists = []
            for idx in noun_idx:
                comb_for_idx = [prod for prod in product([idx], modifiers)]
                comb_for_idx_lists.append(comb_for_idx)
            
            # create all combinations replacements
            noun_replace_combs = [n_r_c for n_r_c in product(*comb_for_idx_lists)]

            for noun_replace_comb in noun_replace_combs:
                new_tags = tags3[:]
                for replacement_tuple in noun_replace_comb:
                    if replacement_tuple[1] == False:
                        pass
                    else: 
                        new_tags[replacement_tuple[0]] = replacement_tuple[1]

                NP_tags_modified.append(new_tags)

        return NP_tags_modified
    
    def create_NP_tagsets(self) -> list:
        """
        Function that creates combinations of spacy POS tags to identify short noun phrases.

        nouns: List of lists. Each sublist is a simple combination of POS tags for NPs.
        det_and_poss: List of lists. Each sublists is a determiner/possessive p. + noun combination.
        modifiers: List of lists. Each sublist is a noun modifier + noun combination.

        Note that the various POS tags for nouns, adjectives and adverbs should be replaced by "noun", adj", "adv" in the input lists.
        Use get_detailed_pos_tags() to see which POS tags are considered.  

        returns: List of lists. Each sublists is a POS tag combination for a short noun phrase from input lists. 
        """

        # load the nouns and modifiers that make up the new POS tag combinations
        nouns, det_and_poss, modifiers = self.get_starter_pos_tags()

        # 1. add new combinations by creation of compound nouns ("noun" to "noun + noun")
        NP_tags_nouns = self.create_noun_compounds(nouns)
        NP_tags_nouns_flat = [self.flatten_list(tag_comb) for tag_comb in NP_tags_nouns] # flatten the tagsets

        # next steps will only be applied to single nouns or the first nouns before another noun (see docstring of get_noun_idx_list())
        
        # 2. add new combinations by modifying nouns to "DET + noun" or "PRP$ + noun" (determiners (incl. "no"), possessive pronouns)
        NP_tags_det_poss = self.create_det_poss_mods(NP_tags_nouns_flat, det_and_poss)
        NP_tags_det_poss_flat = [self.flatten_list(tag_comb2) for tag_comb2 in NP_tags_det_poss]

        # 3. add new combinations by modifying nouns with noun modifiers
        NP_tags_modified = self.create_modifier_mod(NP_tags_det_poss_flat, modifiers)
        NP_tags_modified_flat = [self.flatten_list(tag_comb3) for tag_comb3 in NP_tags_modified]

        return NP_tags_modified_flat
    
    def extract_NP_ADEs(self, NP_tag_comb: list, n=100) -> list:
        """Extract n ADEs from ADE list that match the POS tag combinations in NP_tag_comb.

        Args:
            NP_tag_comb (List): List of list of str with all desired POS tag combinations. Used to filter the ADEs.
            n (int, optional): Number of matching ADEs that should be extracted. Defaults to 100. If set to -1, will extract all matching ADEs. 

        Returns:
            List: All extracted matching unique ADEs.
        """

        # load detailed noun, adj, adv spacy POS tags
        noun_tags_detailed, adj_tags_detailed, adv_tags_detailed = self.get_detailed_pos_tags_nouns_adj_adv()

        n_all_ades = len(self.ade_list) # amount of all (unique) ades
        ades_idx = [i for i in range(n_all_ades)]
        np.random.shuffle(ades_idx) # shuffles indices in-place

        if n == -1:
            n_ades = n_all_ades +1
        else:
            n_ades = n

        ade_nouns = set()

        for idx in ades_idx:
            # get POS tags
            sample = self.spacy_pipe(self.ade_list[idx])
            tags_original = [token.tag_ for token in sample]
            
            # convert POS tags to adj/noun
            tags = []
            for tag in tags_original:
                if tag in noun_tags_detailed:
                    tags.append("noun")
                elif tag in adj_tags_detailed:
                    tags.append("adj")
                elif tag in adv_tags_detailed:
                    tags.append("adv")
                else:
                    tags.append(tag)

            # check if tags of sample match with desired NP tags, add to noun list and strip starting or trailing whitespaces
            if tags in NP_tag_comb:
                ade_nouns.add(self.ade_list[idx].strip())

            # stop the loop when desired number of matching ADEs are found
            if len(ade_nouns) >= n_ades:
                break
        
        return list(ade_nouns)
    
    def count_matching_ADEs(self, NP_tag_comb: list):
        """Counts Extracted ADEs from ADE list that match the POS tag combinations in NP_tag_comb.

        Args:
            NP_tag_comb (List): List of list of str with all desired POS tag combinations. Used to filter the ADEs.

        Returns:
            Tuple: (number of unique extracted ADEs, number of all unique ADEs).
        """
        
        
        # load detailed noun, adj, adv spacy POS tags
        noun_tags_detailed, adj_tags_detailed, adv_tags_detailed = self.get_detailed_pos_tags_nouns_adj_adv()

        n_all_ades = len(list(set(self.ade_list))) # amount of all unique ADEs
        ades_idx = [i for i in range(n_all_ades)]
        np.random.shuffle(ades_idx) # shuffles indices in-place

        ade_nouns = set()
        ade_other = set()

        for idx in ades_idx:
            # get POS tags
            sample = self.spacy_pipe(self.ade_list[idx])
            tags_original = [token.tag_ for token in sample]
            
            # convert POS tags to adj/noun
            tags = []
            for tag in tags_original:
                if tag in noun_tags_detailed:
                    tags.append("noun")
                elif tag in adj_tags_detailed:
                    tags.append("adj")
                elif tag in adv_tags_detailed:
                    tags.append("adv")
                else:
                    tags.append(tag)

            # check if tags of sample match with desired NP tags, strip starting and trailing whitespace
            if tags in NP_tag_comb:
                ade_nouns.add(self.ade_list[idx].strip())
            else:
                ade_other.add(self.ade_list[idx]) 

        ade_nouns = list(ade_nouns)
        ade_other = list(ade_other)

        print("Number of ADEs as NP:", len(ade_nouns))
        print("Number of not extracted ADEs:", len(ade_other))
        print("total number of ADEs:", n_all_ades)
        print("retrieved %: {:.2f}".format((len(ade_nouns)/n_all_ades)*100))

        return ade_nouns, ade_other

    def save_list(self, inp, output_filename: str) -> None:
        """Saves input (inp) to pickle file.

        Args:
            inp: Python object to save.
            output_filename (Str): Name of output file to save pickled object.
        """

        with open(output_filename, "wb") as fp:   
            pickle.dump(inp, fp)

