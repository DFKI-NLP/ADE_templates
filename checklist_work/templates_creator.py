import re
from itertools import product


class TempOrderTemplates():
    """Class for constructing templates that test the temporal order capability.
    Template Types:
        - "positive": Temporal order describes that ADE occurred after drug intake. True label is 1/ADE.
        - "negative": Temporal order describes that ADE occurred before drug intake. True label is 0/ no ADE.

        - standard: Standard temporal order template.
        - single_time_entity: Template contains one placeholder for a time entitiy ("2 weeks").
        - relation_time_entity: Template contains two time entity placeholders.
    
    """
    
    def __init__(self) -> None:
        pass

    def _basic_negative_templates(self, use_time_entity, use_relational_time_entity):
        if use_time_entity:
            if use_relational_time_entity:
                temporal_order_templates = [
                    "I was experiencing {ade} for {time_entity_l}, {time_entity_s} ago I started taking {drug}.",
                    "{time_entity_l} ago I started experiencing {ade}, I have been taking {drug} for {time_entity_s}.",
                    "{time_entity_l} ago I started experiencing {ade}, {time_entity_s} ago I started taking {drug}."
                ]
            else:
                temporal_order_templates = [ 
                    "I was experiencing {ade} for {time_entity}, now I started taking {drug}.",
                    "{time_entity} ago I started experiencing {ade}, now I started taking {drug}.",
                ]
        else:
            temporal_order_templates = [
                "I took {drug} after experiencing {ade}.",
                "After experiencing {ade}, I took {drug}.",
                "I experienced {ade} before taking {drug}.",
                "I experienced {ade} ahead of taking {drug}.",
                "Before taking {drug}, I experienced {ade}.", 
                "Ahead of taking {drug}, I experienced {ade}.",
                "When I started taking {drug}, I was already experiencing {ade}.",
                "I was already experiencing {ade}, when I started taking {drug}.", 
                "I was experiencing {ade}, then I took {drug}.", 
                "I was experiencing {ade}, but it was not due to the {drug} because I had had {ade} before.",  
                "Should I start taking {drug} when I'm already experiencing {ade}?", 
                "I'm unsure if I should start taking {drug}, when I'm already experiencing {ade}.", 
                "Prior to taking {drug}, I was experiencing {ade}.", 
                "I was experiencing {ade}, prior to taking {drug}."
            ]

        return temporal_order_templates
    
    def _basic_positive_templates(self, use_time_entity, use_relational_time_entity):

        if use_time_entity:
            if use_relational_time_entity:
                temporal_order_templates = [
                    "I was experiencing {ade} for {time_entity_s}, {time_entity_l} ago I started taking {drug}.",
                    "{time_entity_l} ago I started taking {drug}, I have been experiencing {ade} for {time_entity_s}.", 
                    "{time_entity_s} ago I started experiencing {ade}, {time_entity_l} ago I started taking {drug}."
                ]
            else:
                temporal_order_templates = [
                    "I was taking {drug} for {time_entity}, now I started experiencing {ade}.",
                    "{time_entity} ago I started taking {drug}, now I started experiencing {ade}.",
                ]
        else:
            temporal_order_templates = [
            "I took {drug} before experiencing {ade}.",
            "After taking {drug}, I experienced {ade}.",
            "I took {drug} before I experienced {ade}.",
            "I took {drug} ahead of experiencing {ade}.",
            "Before experiencing {ade}, I took {drug}.",
            "Ahead of experiencing {ade}, I took {drug}.",
            "When I started experiencing {ade}, I was already taking {drug}.",
            "I was already taking {drug}, when I started experiencing {ade}.",
            "I was taking {drug}, when I experienced {ade}.",
            "I was experiencing {ade}, and I am sure it was due to the {drug} because I had never had {ade} before.",
            "Should I stop taking {drug} when I'm experiencing {ade}?",
            "I'm unsure if I should stop taking {drug}, when I'm experiencing {ade}.",
            ]

        return temporal_order_templates

    def _modify_templates_took_taking_synonyms(self, temporal_order_templates):
        temporal_order_templates_mod = []

        pattern_1_p = re.compile("took")
        pattern_2_n = re.compile("i was taking|was already|ahead of|started taking|prior to taking|start taking|stop taking|been taking")
        pattern_3_p = re.compile("started taking|prior to taking")
        pattern_3_n = re.compile("start taking|was taking")
        pattern_4_p = re.compile("was taking|was already taking")

        for s in temporal_order_templates:
            if re.findall(pattern_1_p, s.lower()):
                temporal_order_templates_mod.append(s.replace("took", "was on"))
                temporal_order_templates_mod.append(s.replace("took", "was treated with"))
                temporal_order_templates_mod.append(s.replace("took", "was put on"))
                temporal_order_templates_mod.append(s.replace("took", "was medicated with"))
                temporal_order_templates_mod.append(s.replace("took", "started taking"))

            elif "taking" in s:
                if not re.findall(pattern_2_n, s.lower()):
                    temporal_order_templates_mod.append(s.replace("taking", "I was treated with"))
                    temporal_order_templates_mod.append(s.replace("taking", "I was put on"))
                    temporal_order_templates_mod.append(s.replace("taking", "I was medicated with"))
                    temporal_order_templates_mod.append(s.replace("taking", "I started taking"))
                elif re.findall(pattern_3_p, s.lower()) and not re.findall(pattern_3_n, s.lower()):
                    temporal_order_templates_mod.append(s.replace("taking", "being treated with"))
                    temporal_order_templates_mod.append(s.replace("taking", "being on"))
                    temporal_order_templates_mod.append(s.replace("taking", "being medicated with"))
                elif re.findall(pattern_4_p, s.lower()):
                    temporal_order_templates_mod.append(s.replace("taking", "treated with"))
                    temporal_order_templates_mod.append(s.replace("taking", "on"))
                    temporal_order_templates_mod.append(s.replace("taking", "medicated with"))


        return list(set(temporal_order_templates_mod)) # these are only the modified tests, not including the old tests 
    
    def _modify_templates_experience_synonyms(self, temporal_order_templates):
        
        temporal_order_templates_mod = []
        for s in temporal_order_templates:
            if "experienced" in s:
                temporal_order_templates_mod.append(s.replace("experienced", "had"))
                temporal_order_templates_mod.append(s.replace("experienced", "suffered from"))
                temporal_order_templates_mod.append(s.replace("experienced", "encountered"))
                temporal_order_templates_mod.append(s.replace("experienced", "endured"))
                temporal_order_templates_mod.append(s.replace("experienced", "lived through"))
            elif "experiencing" in s:
                temporal_order_templates_mod.append(s.replace("experiencing", "having"))
                temporal_order_templates_mod.append(s.replace("experiencing", "suffering from"))
                temporal_order_templates_mod.append(s.replace("experiencing", "encountering"))
                temporal_order_templates_mod.append(s.replace("experiencing", "enduring"))
                temporal_order_templates_mod.append(s.replace("experiencing", "living through"))

        return list(set(temporal_order_templates_mod)) # these are only the modified tests, not including the old tests 
    
    def create_all_templates(self, template_type, use_time_entity=False, use_relational_time_entity=False):
        """Function that creates all temporal order templates of the specified type.

        Args:
            template_type (str): Specifiy template type to create either "negative" (no ADE) or "positive" (ADE) templates.
            use_time_entity (bool, optional): Specify whether to create templates that have time entity placeholders. Defaults to False.
            use_relational_time_entity (bool, optional): If use_time_entity is True, use this argument to specify whether to 
                create placeholders with relational time entities. Defaults to False.

        Raises:
            Exception: Raised when template_type argument is neither "negative" nor "positive".

        Returns:
            Dict: Created templates of specified type with basic templates as keys and template variations (incl. basic template) as values.
        """

        # if use_time_enitiy is False, then only templates not including time_entity are created
        # if use_time_enitiy is True and use_relational_time_entity is False, then only templates including single time_entity are created
        # if use_time_enitiy is True and use_relational_time_entity is True, then only templates including relational time_entity are created

        if use_relational_time_entity:
            assert use_time_entity, "If relational time entities are expected, then use_relational_time_entity must be set to True in args."

        if template_type == "positive":
            basic_templates = self._basic_positive_templates(use_time_entity=use_time_entity, use_relational_time_entity=use_relational_time_entity)
        elif template_type == "negative":
            basic_templates = self._basic_negative_templates(use_time_entity=use_time_entity, use_relational_time_entity=use_relational_time_entity)
        else:
            raise Exception("argument 'template_type' must be either 'positive' or 'negative'")
        
        # create variations of each basic template and store variations as a dictionary
        templates = dict()
        for basic_t in basic_templates:
            template_variations = [basic_t] # basic template is part of the variations
            template_variations += self._modify_templates_took_taking_synonyms(template_variations)
            template_variations += self._modify_templates_experience_synonyms(template_variations)
            templates[basic_t] = sorted(template_variations)

        return templates
    

class PosSentimentTemplates():
    """Class for constructing templates that test the positive sentiment and beneficial effects capability.
    Template Types:
        - Positive sentiment templates express a positive sentiment despite describing an ADE. 
        - Beneficial effects templates describe secondary effects of a drug that benefit the drug user despite not being the reason for the drug intake.
        - Non-beneficial effects are templates that mention an effect that could be beneficial but is not in the given context.
    """

    def __init__(self) -> None:
        pass

    def _basic_pos_sentiment_templates(self):
        """Create the two basic parts of the templates and the connectors.
        Basic parts are part_a (description of ADE) and part_b (description of positive sentiment).
        Two differnt types of connectors exist for the two different setups of the template parts:
            - version 1.1: part A. connector1, part B. (2 sentences)
            - version 1.2.: part A, connector1 part B. (1 sentence)
            - version 2.1: part B. connector2, part A. (2 sentences)
            - version 2.2.: part B, connector2 part A. (1 sentence) 
        The connector for version 1 may be left out: v1.1.: part A. part B. / v1.2.: part A, part B.
        """

        part_a = [
            "I'm taking {drug} and experiencing {ade}",
            "{drug} gives me {ade}",
            "I always have {ade} when I'm on {drug}",
            "for me, {ade} is a side-effect of {drug}"
        ]

        part_b = [
            "I am happy my symptoms have reduced",
            "I feel much better in general than without {drug}",
            "my quality of life has increased dramatically",
            "I haven't felt this good in years",
            "I am happier than before taking {drug}",
            "it is okay because {ade} is not strong",
            "it's okay because {ade} doesn't stop me from enjoying life",
            "it's okay because I can ignore {ade}",
            "I like {drug} because {ade} is less severe now compared to my previous medication" 
        ]

        connectors_v1 = [
            "but",
            "still",
            "yet",
            ""
        ]

        connectors_v2 = [
            "even though",
            "although",
            "but"
        ]

        return part_a, part_b, connectors_v1, connectors_v2
    
    def _compose_templates_pos_sentiment_v1(self, template_part_a, template_part_b, connectors_v1):
        """Creates full templates for positive sentiment (not beneficial effect) for version 1:
            - version 1.1: part A. connector, part B. (2 sentences)
            - version 1.2.: part A, connector part B. (1 sentence)
        The connector for version 1 may be left out. v1.1.: part A. part B. / v1.2.: part A, part B.

        Args:
            templates_part_a (str): part_a template.
            templates_part_b (str): part_b template.
            connectors_v1 (list): list of valid connectors for version 1.
        
        Returns:
            List: All templates of versons 1.1. and 1.2.
    
        """
        # version 1.1: part A. connector, part B. (2 sentences)
        templates_v11 = []
        # version 1.2.: part A, connector part B. (1 sentence)
        templates_v12 = []

        t_a = str(template_part_a[0].upper() + template_part_a[1:]) # capitalize part_a
        t_b = template_part_b
        for connector in connectors_v1:
            if connector == "": # no connector
                t_b_modified = str(t_b[0].upper() + t_b[1:])
                templates_v11.append(str(t_a + ". " + t_b_modified + "."))
                templates_v12.append(str(t_a + ", " + t_b + "."))
            else:
                templates_v11.append(str(t_a + ". " + connector.capitalize() + ", " + t_b + "."))
                templates_v12.append(str(t_a + ", " + connector + " " + t_b + "."))
        
        return templates_v11 + templates_v12
    
    def _compose_templates_pos_sentiment_v2(self, template_part_a, template_part_b, connectors_v2):
        """Creates full templates for positive sentiment (not beneficial effect) for version 2:
            - version 2.1: part B. connector, part A. (2 sentences)
            - version 2.2.: part B, connector part A. (1 sentence)

        Args:
            templates_part_a (str): part_a template.
            templates_part_b (str): part_b template.
            connectors_v2 (list): list of valid connectors for version 2.
        
        Returns:
            List: All templates of versons 2.1. and 2.2.
        """
        # version 2.1: part B. connector, part A. (2 sentences)
        templates_v21 = []
        # version 2.2.: part B, connector part A. (1 sentence)
        templates_v22 = []

        t_b = str(template_part_b[0].upper() + template_part_b[1:]) # capitalize part_b
        t_a = template_part_a
        for connector in connectors_v2:
            templates_v21.append(str(t_b + ". " + connector.capitalize() + " " + t_a + "."))
            templates_v22.append(str(t_b + ", " + connector + " " + t_a + "."))
        
        return templates_v21 + templates_v22
    
    def _beneficial_effect_templates(self, part_a):
        """Create all templates for beneficial effect statements.

        Templates consist of part_a, which is shared with the positive sentiment templates, and part_b,
        which is unique to the beneficial effect templates. The beneficial effects are fixed and there is
        only one matching part_b for every beneficial effect. 
        
        For each beneficial effect template, there is also a non-beneficial counterpart template,
        which desribes a secondary effect of a drug that is considered in a negative way by the drug user. 
        
        The ADE placeholders are filled-in with the beneficial effects.

        Args:
            part_a (list): part_a templates.

        Returns:
            dict: A Dictionary where keys are "beneficial_effect" and "non_beneficial_effect" and values are dictionaries of templates with beneficial effect types as keys.
        """

        # Only one version exists: 
        # version 1.1: part A. part B. (2 sentences)

        beneficial_effects_part_b = {
            "weight loss": "I'm happy because I was trying to lose weight anyway",
            "weight gain": "I'm happy because I was trying to gain weight anyway",
            "sleepiness": "I'm happy because I had trouble sleeping at night before",
            "decreased need for sleep": "I'm happy because I had trouble staying awake during the day before",
            "loss of appetite": "I'm happy because I was overeating before",
            "increased appetite": "I'm happy because I was suffering from loss of appetite before"
            }
        
        non_beneficial_effects_part_b = {
            "weight loss": "it's a problem because I am already underweight",
            "weight gain": "it's a problem because I am already overweight",
            "sleepiness": "it's a problem because I have trouble staying awake during the day anyway",
            "decreased need for sleep": "it's a problem because I have trouble sleeping at night anyway",
            "loss of appetite": "it's a problem because I was suffering from loss of appetite before",
            "increased appetite": "it's a problem because I was overeating before"
            }

        # beneficial effect templates are stored by keys (the beneficial effect)
        b_effects_templates_dict = {}
        non_b_effects_templates_dict = {}
        
        # loop over beneficial effects to combine the template parts and fill in the ade placeholder
        for b_effect in beneficial_effects_part_b.keys():
            b_effect_templates = []
            non_b_effect_templates = []
            
            for t_a in part_a: 
                t_a_effect = t_a.replace("{ade}", b_effect) # replace ade placeholder with specific beneficial effect
                
                # for version 1.1.
                t_a_effect = str(t_a_effect[0].upper() + t_a_effect[1:])

                # beneficial effects sentence
                t_b = beneficial_effects_part_b[b_effect]
                # for version 1.1. 
                t_b = str(t_b[0].upper() + t_b[1:])
                b_effect_templates.append(str(t_a_effect + ". " +  t_b + "."))

                # non-beneficial effect sentence
                t_b = non_beneficial_effects_part_b[b_effect]
                # for version 1.1. 
                t_b = str(t_b[0].upper() + t_b[1:])
                non_b_effect_templates.append(str(t_a_effect + ". " +  t_b + "."))
            
            b_effects_templates_dict[b_effect] = b_effect_templates
            non_b_effects_templates_dict[b_effect] = non_b_effect_templates
        
        return {"beneficial_effect": b_effects_templates_dict, "non_beneficial_effect": non_b_effects_templates_dict}
    
    def create_all_templates(self, beneficial_effects=False):
        """Function that creates all positive sentiment or beneficial effects templates.

        Args:
            beneficial_effects (bool, optional): Set to True to create beneficial effects templates instead of positive sentiment templates. Defaults to False.

        Returns:
            Dict: If positive sentiment templates, output is a dictionary with part_a-part_b combinations as keys and their template variations as values.
                If beneficial effects templates, output is a dictionary with beneficial effects/non-b. effects as keys and dictionaries of beneficial effect types (keys) and templates (values) as values.
        """

        # get template components
        part_a, part_b, connectors_v1, connectors_v2 = self._basic_pos_sentiment_templates()

        if beneficial_effects: # create templates that have beneficial effects instead of ADEs
            b_effects_templates_all_dict = self._beneficial_effect_templates(part_a)

            return b_effects_templates_all_dict
        
        else: # create positive sentiment ADE templates
            basic_2_parts_generator = product(part_a, part_b) # get part_a part_b permutations (cartesian product)

            templates = dict()
            for basic_part in basic_2_parts_generator:
                templates_v1 = self._compose_templates_pos_sentiment_v1(template_part_a=basic_part[0], template_part_b=basic_part[1], connectors_v1=connectors_v1)
                templates_v2 = self._compose_templates_pos_sentiment_v2(template_part_a=basic_part[0], template_part_b=basic_part[1], connectors_v2=connectors_v2)
                templates[basic_part] = templates_v1 + templates_v2

            return templates

class NegationTemplates():
    """Class for constructing templates that test the negation capability.
    Template types:
            - "negative": Templates contain an ADE that is negated. True label is therefore 0/no ADE.
            - "positive": Templates contain an ADE and some negation without negating the ADE. True label is 1/ADE.
    """
    
    def __init__(self) -> None:
        pass

    def _basic_negative_templates(self):
        """Creates basic negative (true label is 0/no ADE) templates where the ADE is negated."""

        templates = [
            "I took {drug} and did not experience {ade}.",
            "I've never experienced {ade} when taking {drug}.",
            "For me, {drug} didn't cause {ade}.",
            "No signs of {ade} when taking {drug}.",
            "I'm lucky I wasn't experiencing {ade} when taking {drug}.",
            "I am taking {drug} without experiencing {ade}.",
            "Taking {drug} didn't cause {ade} for me.",
            "Never experienced {ade} caused by {drug}.",
            "Ever since I started taking {drug}, I've had no signs of {ade}.",
            "For me, {ade} is not a side-effect of {drug}.",
            "For me, {drug}-related {ade} has never been an issue."
        ]

        return templates
    
    def _basic_positive_templates(self):
        """Creates basic positive (true label is 1/ADE) templates that contain some form of negation
        without negating the ADE."""

        templates = [
            "That's not true, I took {drug} and experienced {ade}.",
            "I have never not experienced {ade} when I'm on {drug}.",
            "I don't like {drug}, it gives me {ade}.",
            "I don't like my doctor who gave me {drug}, now I'm experiencing {ade}."  
        ]

        return templates
    
    def _modify_templates_experience_synonyms(self, basic_templates):
        
        templates_mod = []
        for s in basic_templates:
            if "experienced" in s:
                templates_mod.append(s.replace("experienced", "had"))
                templates_mod.append(s.replace("experienced", "suffered from"))
                templates_mod.append(s.replace("experienced", "encountered"))
                templates_mod.append(s.replace("experienced", "endured"))
                templates_mod.append(s.replace("experienced", "lived through"))
            elif "experiencing" in s:
                templates_mod.append(s.replace("experiencing", "having"))
                templates_mod.append(s.replace("experiencing", "suffering from"))
                templates_mod.append(s.replace("experiencing", "encountering"))
                templates_mod.append(s.replace("experiencing", "enduring"))
                templates_mod.append(s.replace("experiencing", "living through"))

        return list(set(templates_mod)) # these are only the modified tests, not including the old tests 
   
    
    def _modify_templates_took_taking_synonyms(self, basic_templates):
        
        templates_mod = []

        pattern_1_p = re.compile("took")
        pattern_2_p = re.compile("taking")
        pattern_2_n = re.compile("started taking|i'm taking|i am taking")

        for s in basic_templates:
            if re.findall(pattern_1_p, s.lower()):
                templates_mod.append(s.replace("took", "was on"))
                templates_mod.append(s.replace("took", "was treated with"))
                templates_mod.append(s.replace("took", "was put on"))
                templates_mod.append(s.replace("took", "was medicated with"))
                templates_mod.append(s.replace("took", "started taking"))

            elif re.findall(pattern_2_p, s.lower()):
                if not re.findall(pattern_2_n, s.lower()) and not s.lower().startswith("taking"):
                    templates_mod.append(s.replace("taking", "I was treated with"))
                    templates_mod.append(s.replace("taking", "I was put on"))
                    templates_mod.append(s.replace("taking", "I was medicated with"))
                    templates_mod.append(s.replace("taking", "I started taking"))

        return list(set(templates_mod)) # these are only the modified tests, not including the old tests 


    def create_all_templates(self, template_type):
        """Function that creates all negation templates of the specified type.

        Template types:
            - "negative": Templates contain an ADE that is negated. True label is therefore 0/no ADE.
            - "positive": Templates contain an ADE and some negation without negating the ADE. True label is 1/ADE.

        Args:
            template_type (str): Specifiy template type to create either "negative" (no ADE) or "positive" (ADE) templates.

        Raises:
            Exception: Raised when template_type argument is neither "negative" nor "positive".

        Returns:
            Dict: Dictionary of created templates where the keys are basic templates and values are variations of the basic template incl. the basic template.
        """

        if template_type == "positive":
            basic_templates = self._basic_positive_templates()
        elif template_type == "negative":
            basic_templates = self._basic_negative_templates()
        else:
            raise Exception("argument 'template_type' must be either 'positive' or 'negative'")
        
        # create variations of each basic template and store variations as a dictionary
        templates = dict()
        for basic_t in basic_templates:
            template_variations = [basic_t] # basic template is part of the variations
            template_variations += self._modify_templates_took_taking_synonyms(template_variations)
            template_variations += self._modify_templates_experience_synonyms(template_variations)
            templates[basic_t] = sorted(template_variations)

        return templates

    




