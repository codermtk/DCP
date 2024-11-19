import re
from openai import OpenAI
import json
import inflect


OPENAI_API_KEY = "" 
client_openai = OpenAI(api_key=OPENAI_API_KEY)
model="gpt-4o"

class ConceptualDeconstruction:
    """
   Class to apply FAD to a concept or field of knowledge.
    """
    def __init__(self, initial_concept, max_depth, num_concepts):

        """
        Initialises the class .

        :param initial_concept: The concept or field of knowledge to be deconstructed.
        :param max_depth: The maximum depth or maximum number of subdivisions to be applied by the algorithm.
        :param num_concepts: The maximum number of sub-concepts or sub-fields that the algorithm can extract per concept or field of knowledge..
        """

        self.initial_concept = initial_concept.lower()
        self.max_depth = max_depth
        self.num_concepts = num_concepts
        self.pending_list = [f"{self.initial_concept}_0.1.0"]
        self.analysed_list = []
        self.waitlist = []
        self.p = inflect.engine()
        self.no_return = []
        self.conceptual_quarks = []
        self.conceptual_loops = []
        self.branches_concepts = {self.initial_concept: []}
        self.branch_history = []

    def normalise_concept(self, concept):

        """
        It converts the concept or field of knowledge into a format that is easier for the algorithm to process, and does so by converting everything to lower case.
        """

        return concept.lower()

    def are_similar_custom(self, concept1, concept2, threshold=0.75):

        """
        This function checks if two concepts are written the same, from left to right, and at a % indicating the threshold.

        :param concept1: The first concept.
        :param concept2: The second concept.
        :param threshold: The character-by-character match % the function is looking to detect, starting from the left.
        """

        concept1 = concept1.lower()
        concept2 = concept2.lower()
        

        len1 = len(concept1)
        len2 = len(concept2)
        min_len = min(len1, len2)
        N = max(1, round(min_len * threshold))  


        sub_concept1 = concept1[:N]
        sub_concept2 = concept2[:N]

        return sub_concept1 == sub_concept2

    def extract_loop(self, branch, repeated_concept):

        """
        This function extracts concept loops from branches.

        :param branch: The branch from which a conceptual loop is to be drawn.
        :param repeated_concept: The name of the concept or field of knowledge that is repeated.
        """

        index = [i for i, c in enumerate(branch) if c == repeated_concept]
        if len(index) >= 2:
            start = index[0]
            end = index[-1]
            return branch[start:end + 1]
        return []



    def decompose_concept(self, concept, level, context):

        """
        This function decomposes a concept or knowledge field into n maximum number of :param num_concepts: concepts or knowledge subfields.
        It does this using the OpenAI API and the model that the user selects.
          

        :param concept: The concept to be decomposed.
        :param level: The depth level at which the concept to be decomposed is located.
        :param context: The context is all the concepts from which the concept to be decomposed comes from, i.e. the branch.
        """

        concept = concept.lower()  
       
        print(f"Deconstructing concept: '{concept}', level: {level}, context: '{context}'")

        messages = [
            {
                "role": "system",
                "content": (
                    f"You're an expert on definitions."
                    f"You will be presented with a chain of subdivisions and your job is to determine the following two criteria:"
                    f"1.Whether the chain follows a clear decreasing order (Example done right: Ancient Greece -> Greek Philosophy -> Plato) (Example done wrong: truth -> fact -> evidence -> observation -> analysis"
                    f"2. If all terms in the chain are related by exact definition."
                    f"If one of these two criteria is not met for the last term added to the chain you must return in JSON format the word “empty” for all values"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The last term added to the chain is: ‘{concept}’. "
                    f"The chain of subdivisions that has led to this term is: ‘{context}’.."
                    f"Please analyse whether the term can be subdivided into other {self.num_concepts} terms."                  
                    f"The format should be: {{‘concept_1’: ‘subdivision1’, ‘concept_2’: ‘subdivision2’, ...}} "
                    f"You can return fewer terms than the {self.num_concepts}, but you must return the word ‘empty’ on the extra term slots."
                    f"I've been testing you and you never put ‘empty’, remember to put ‘empty’ if you see a term in the string does not relate 100% to the original one."  
                    f"If you see the chain empty it is because it is the first term, try not to return empty in that case."
                ),
            },
        ] 


        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "concept_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        f"concept_{i + 1}": {
                            "description": f"The most relevant noun {i + 1} taken from the definition.",
                            "type": "string"
                        } for i in range(self.num_concepts)
                    },
                    "required": [f"concept_{i + 1}" for i in range(min(self.num_concepts, 1))],
                    "additionalProperties": False
                }
            }
        }

        try:

            response = client_openai.chat.completions.create(
                model=model,  
                messages=messages,
                response_format=response_format

            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:

            return f"Error processing the definition with OpenAI: {str(e)}"

    
    def process(self):

        """
        This function is in charge of implementing DCP and coordinating the other functions in this file.
        """

        while self.pending_list:

            actual_concept = self.pending_list.pop(0)
            concept_name, rest = actual_concept.rsplit('_', 1)
            level, j, k = [int(x) for x in rest.split('.')]
            concept_name = concept_name.lower()

            concept_name = self.normalise_concept(concept_name)

            current_branch = self.branches_concepts.get(actual_concept, [])

            context = " -> ".join(current_branch)


            found_loop = False
            for ancestor in current_branch:

                if self.are_similar_custom(concept_name, ancestor):
                    loop = self.extract_loop(current_branch, ancestor) + [concept_name]
                    self.conceptual_loops.append(loop)
                    found_loop = True
                    break  

            if found_loop:

                continue


            subconcepts = self.decompose_concept(concept_name, level, context)
            print(f"Sub-concepts received for '{concept_name}': {subconcepts}")

            if 'Error' in subconcepts:

                self.no_return.append(actual_concept)
                self.conceptual_quarks.append(current_branch + [concept_name])
                continue


            new_branch = current_branch + [concept_name]
            try:
                subconcepts_dict = json.loads(subconcepts.replace("'", '"'))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"API Response: {subconcepts}")
                self.conceptual_quarks.append(new_branch)
                continue

            if not subconcepts_dict:
                self.conceptual_quarks.append(new_branch)
                continue

            has_subconcepts = False
            for idx, subconcept in enumerate(subconcepts_dict.values(), start=1):
                subconcept = subconcept.lower()
                subconcept = self.normalise_concept(subconcept)
                if subconcept == 'empty':
                    continue  
                else:
                    subconcept_id = f"{subconcept}_{level + 1}.{idx}.{j}"
                    similar_to_ancestor = False
                    repeated_ancestor = None  
                    for ancestor in new_branch:
                        if self.are_similar_custom(subconcept, ancestor):
                            similar_to_ancestor = True
                            repeated_ancestor = ancestor
                            break
                    if similar_to_ancestor:
                        loop = self.extract_loop(new_branch, repeated_ancestor) + [subconcept]
                        self.conceptual_loops.append(loop)
                        continue
                    else:
                        has_subconcepts = True
                        if level + 1 <= self.max_depth:
                            self.pending_list.append(subconcept_id)
                        self.branches_concepts[subconcept_id] = new_branch
                        self.branch_history.append(new_branch + [subconcept])

            if not has_subconcepts:
                self.conceptual_quarks.append(new_branch)

            self.analysed_list.append(actual_concept)


        flat_loops = set()
        for loop in self.conceptual_loops:
            flat_loops.update(loop)

        return {

            "Analysed Concepts": self.analysed_list,
            "Conceptual Quarks": self.conceptual_quarks,
            "Conceptual Loops": list(self.conceptual_loops),
            "Flattened Loops": list(flat_loops),
            "Branch History": self.branch_history

        }
    
def main():
    """
    Main function of the script.
    If only the algorithm without the UI is to be used, the initial parameters must be indicated here.
    From the concept or knowledge field to deconstruct :initial_concept:, to the maximum number of subdivisions :max_depth: 
    to the maximum number of subconcepts or subfields of knowledge into which the algorithm can deconstruct each concept or field of knowledge :num_concepts:.
    If you intend to use the UI, these parameters are set in UIDCP.py, not here.
    """

    initial_concept = ""
    max_depth = 6
    num_concepts = 5
    deconstruction = ConceptualDeconstruction(initial_concept, max_depth, num_concepts)
    result = deconstruction.process()
    print("Outcome of Conceptual Deconstruction:")
    print(result)

if __name__ == "__main__":
    main()