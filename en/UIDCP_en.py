import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, simpledialog
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import re
import threading
import numpy as np
from openai import OpenAI
import openai  

from DCP_en import ConceptualDeconstruction

OPENAI_API_KEY = "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" 
client_openai = OpenAI(api_key=OPENAI_API_KEY)


def hierarchy_pos(G, root, node_levels, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        
        """
        This function generates hierarchical positions for the nodes.
        :param G: The directed network (must be a tree).
        :param root: Root node of the tree.
        :param node_levels: Dictionary with the different levels of the nodes.
        param width: Width of the tree.
        :param vert_gap: Vertical space between levels.
        param vert_loc: Vertical position of the root node.
        param xcenter: Horizontal centre position of the root node.
        :return: The dictionary with the positions {node: (x, y)}.
        """

        def _hierarchy_pos(G, node, left, right, vert_gap, vert_loc, pos, parent=None):

            children = list(G.successors(node))
            if not children:
                pos[node] = ((left + right) / 2, vert_loc)
                return pos
            total_leaves = sum([count_leaves(G, child) for child in children])
            current_left = left
            for child in children:
                num_leaves = count_leaves(G, child)
                child_width = (num_leaves / total_leaves) * (right - left)
                child_right = current_left + child_width
                pos = _hierarchy_pos(G, child, current_left, child_right, vert_gap, vert_loc - vert_gap, pos, node)
                current_left += child_width
            child_positions = [pos[child][0] for child in children]
            pos[node] = (sum(child_positions) / len(child_positions), vert_loc)
            return pos

        def count_leaves(G, node):
            """
        Counts the number of leaves descending from any given node.
        :param G: The directed graph.
        :param node: Node from which to count leaves.
        :return: Number of descendant leaves.
        """
            
            children = list(G.successors(node))
            if not children:
                return 1
            return sum([count_leaves(G, child) for child in children])

        pos = {}
        _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, pos)
        return pos

def radial_layout(G, root):

    """
    Generates a 360 degree radial layout for the graph, and assigns different radii according to the depth level of the concept.

    :param G: The directed network.
    :param root: Root node of the network.
    :return: Dictionary of positions {node: (x, y)}.

    """

    pos = {}
    levels = nx.single_source_shortest_path_length(G, root)
    max_level = max(levels.values()) if levels else 1

    for node, level in levels.items():
        if level == 0:
            pos[node] = (0, 0)
        else:
            nodes_at_level = [n for n, l in levels.items() if l == level]
            num_nodes = len(nodes_at_level)
            angle_gap = 360 / num_nodes if num_nodes > 0 else 0
            for i, n in enumerate(nodes_at_level):
                angle = angle_gap * i
                radius = level * 3  
                x = radius * np.cos(np.radians(angle))
                y = radius * np.sin(np.radians(angle))
                pos[n] = (x, y)

    return pos

class TreeApp(tk.Tk):

    """
    It is the main class of the application, it creates the graphical interface and visualises the graph resulting from the conceptual deconstruction.
    """
    
    def __init__(self, G, initial_node, node_levels, id_a_concept, conceptual_quarks, nodes_to_colour_red, nodes_to_colour_green):
        super().__init__()
        self.title("Conceptual Deconstruction Visualisation")
        self.geometry("1600x900")
        self.coloring_enabled = True  
        self.active_colors = {
            'initial_node': 'gold',
            'red': 'red',
            'green': 'green',
            'quark': 'green',
            'default': 'skyblue'
        }
        self.layout_mode = 'tree'  
        self.ai_classification = {}  
        self.G = G
        self.initial_node = initial_node
        self.node_levels = node_levels
        self.id_a_concept = id_a_concept
        self.font_size = 10
        self.node_size = 700  
        self.conceptual_quarks = conceptual_quarks
        self.nodes_to_colour_red = nodes_to_colour_red  
        self.nodes_to_colour_green = nodes_to_colour_green  
        self.create_widgets()

    def create_widgets(self):

        """
        Creates the GUI widgets, i.e. the buttons and the drawing area for the network.
        """

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)


        layout_button = ttk.Button(control_frame, text="Change Layout", command=self.toggle_layout)
        layout_button.pack(side=tk.LEFT, padx=5)


        ia_button = ttk.Button(control_frame, text="Analyse Empirical/Conceptual", command=self.start_ai_analysis)
        ia_button.pack(side=tk.LEFT, padx=5)


        normal_colors_button = ttk.Button(control_frame, text="Back to Normal Colours", command=self.reset_ai_coloring)
        normal_colors_button.pack(side=tk.LEFT, padx=5)


        reset_button = ttk.Button(control_frame, text="Analyze New Concept", command=self.prompt_reset)
        reset_button.pack(side=tk.LEFT, padx=5)


        show_quarks_button = ttk.Button(control_frame, text="Show Conceptual Quarks", command=self.show_quarks)
        show_quarks_button.pack(side=tk.LEFT, padx=5)


        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=1)

        self.fig = plt.Figure(figsize=(30, 20))
        self.ax = self.fig.add_subplot(111)


        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


        self.pos = self.get_layout()


        self.draw_graph()


        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def show_quarks(self):

        """
        Displays a list of Concept Quarks found and lets you copy and paste them.
        """

        if not self.conceptual_quarks:
            messagebox.showinfo("Conceptual Quarks", "There are no Concept Quarks to show.")
            return

        quarks_lista = [self.id_a_concept.get(node, node) for node in self.conceptual_quarks]
        quarks_texto = '\n'.join(quarks_lista)


        quarks_window = tk.Toplevel(self)
        quarks_window.title("Conceptual Quarks")
        quarks_window.geometry("600x400")

        text_area = tk.Text(quarks_window, wrap=tk.WORD, font=("Arial", 10))
        text_area.insert(tk.END, quarks_texto)
        text_area.pack(expand=True, fill='both')


        text_area.config(state='normal')



    def prompt_reset(self):

        """
        It prompts the user for new parameters and restarts the entire process with the concept or knowledge field of the user's choice.
        """

        try:
            initial_concept = simpledialog.askstring("Input", "Input the initial concept:", parent=self)
            if initial_concept is None or initial_concept.strip() == "":
                messagebox.showerror("Error", "The initial concept cannot be empty.")
                return

            max_depth = simpledialog.askinteger("Input", "Enter the maximum level of depth:", parent=self, minvalue=1)
            if max_depth is None:
                messagebox.showerror("Error", "The maximum level of depth is mandatory.")
                return

            max_subdivisions = simpledialog.askinteger("Input", "Enter the maximum number of subdivisions per concept:", parent=self, minvalue=1)
            if max_subdivisions is None:
                messagebox.showerror("Error", "You must enter the max number of subdivisions.")
                return


            self.reset_graph(initial_concept, max_depth, max_subdivisions)

            self.pos = self.get_layout()
            self.draw_graph()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while restarting the chart.: {str(e)}")

    def reset_graph(self, initial_concept, max_depth, max_subdivisions):

        """
        Resets the graph with the new parameters.
        param :initial_concept: Root concept or knowledge field for the deconstruction.
        param :max_depth: Maximum depth of the deconstruction, or maximum number of subdivisions.
        param :max_subdivisions: Maximum number of subdivisions per concept.
        """

        deconstruction = ConceptualDeconstruction(initial_concept.lower(), max_depth, max_subdivisions)
        result = deconstruction.process()

        print("The result of Conceptual Deconstruction:")
        print(json.dumps(result, indent=4, ensure_ascii=False))


        G = nx.DiGraph()


        numbered_concept = {}
        node_levels = {}
        id_a_concept = {}


        for concept_id in result.get('Analysed Concepts', []):
            concept_name = concept_id.split('_')[0].lower()
            suffix = concept_id.split('_')[1]
            concept_with_suffix = f"{concept_name}_{suffix}".lower()
            numbered_concept[concept_id] = concept_with_suffix
            id_a_concept[concept_with_suffix] = concept_name

            level = int(suffix.split('.')[0])
            node_levels[concept_with_suffix] = level


        initial_node = f"{initial_concept.lower()}_0.1.0"
        node_levels[initial_node] = 0


        if initial_node not in G:
            G.add_node(initial_node)


        concept_a_ids = {}
        for concept_id, concept_with_suffix in numbered_concept.items():
            concept_name = concept_id.split('_')[0].lower()
            if concept_name not in concept_a_ids:
                concept_a_ids[concept_name] = []
            concept_a_ids[concept_name].append(concept_id)


        numbered_branches = []
        for branch in result.get('Branch History', []):
            numbered_branch = []
            for concept in branch:
                concept = concept.lower()

                possible_ids = concept_a_ids.get(concept, [])
                if possible_ids:

                    concept_id = min(possible_ids, key=lambda cid: node_levels.get(numbered_concept.get(cid), 0))
                    concept_with_suffix = numbered_concept.get(concept_id, concept)
                    numbered_branch.append(concept_with_suffix)
                else:

                    if concept == initial_concept.lower():
                        numbered_branch.append(initial_node)
                    else:
                        numbered_branch.append(concept)
            numbered_branches.append(numbered_branch)


        for numbered_branch in numbered_branches:
            for i in range(len(numbered_branch) - 1):
                parent = numbered_branch[i]
                child = numbered_branch[i + 1]

                if node_levels.get(child, 0) == node_levels.get(parent, 0) + 1:
                    G.add_edge(parent, child)

        print("Network nodes:", G.nodes())
        print("Edges of the graph:", G.edges())


        conceptual_quarks = set()
        for quark in result.get('Conceptual Quarks', []):
            if quark:

                quark_numbered = []
                for concept in quark:
                    concept = concept.lower()
                    possible_ids = concept_a_ids.get(concept, [])
                    if possible_ids:

                        concept_id = max(possible_ids, key=lambda cid: node_levels.get(numbered_concept.get(cid), 0))
                        concept_with_suffix = numbered_concept.get(concept_id, concept)
                        quark_numbered.append(concept_with_suffix)
                    else:
                        quark_numbered.append(concept)

                conceptual_quarks.add(quark_numbered[-1])


        conceptual_loops = set()
        for loop in result.get('Conceptual Loops', []):
            if loop:

                numbered_loop = []
                for concept in loop:
                    concept = concept.lower()
                    possible_ids = concept_a_ids.get(concept, [])
                    if possible_ids:

                        for concept_id in possible_ids:
                            concept_with_suffix = numbered_concept.get(concept_id, concept)
                            numbered_loop.append(concept_with_suffix)
                    else:
                        numbered_loop.append(concept)

                if len(numbered_loop) >= 2:
                    conceptual_loops.update(numbered_loop)


        def are_similar_custom_gui(concept1, concept2, threshold=0.75):

            """
        This function checks if two concepts are written the same, from left to right, and in a % indicating the threshold.
        :param concept1: The first concept.
        :param concept2: The second concept.
        :param threshold: The % of character-by-character match that the function seeks to detect, starting from the left.
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


        def extract_name(concept):
            
            name = concept.split('_')[0]
            name = re.sub(r'\d+', '', name) 
            return name


        ancestors = {node: nx.ancestors(G, node) for node in G.nodes()}


        childless_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]


        nodes_to_colour_green = set()
        nodes_to_colour_red = set()

        for node in childless_nodes:
            node_name = extract_name(id_a_concept.get(node, node))
            for other_node in G.nodes():
                if other_node == node:
                    continue
                other_name = extract_name(id_a_concept.get(other_node, other_node))
                if are_similar_custom_gui(node_name, other_name):

                    if other_node in ancestors[node] or node in ancestors[other_node]:
                        nodes_to_colour_red.add(node)
                        nodes_to_colour_red.add(other_node)


        for node in childless_nodes:
            if node not in nodes_to_colour_red:
                nodes_to_colour_green.add(node)


        self.G = G
        self.initial_node = initial_node
        self.node_levels = node_levels
        self.id_a_concept = id_a_concept
        self.conceptual_quarks = conceptual_quarks
        self.conceptual_loops = conceptual_loops
        self.nodes_to_colour_red = nodes_to_colour_red  
        self.nodes_to_colour_green = nodes_to_colour_green 

    def get_layout(self):

        """
        Calculates the positions of all nodes in the network according to the selected layout.
        :return: Dictionary of positions {node: (x, y)}.
        """

        pos = {}

        reachable_nodes = set(nx.descendants(self.G, self.initial_node)).union({self.initial_node})
        

        subgraph_main = self.G.subgraph(reachable_nodes)
        
        if self.layout_mode == 'tree':

            sub_pos = hierarchy_pos(subgraph_main, self.initial_node, self.node_levels, width=30.0, vert_gap=3.0)
        elif self.layout_mode == 'radial':

            sub_pos = radial_layout(subgraph_main, self.initial_node)
        else:

            sub_pos = nx.spring_layout(subgraph_main, seed=42)

        pos.update(sub_pos)


        missing_nodes = reachable_nodes - set(pos.keys())
        if missing_nodes:
            print(f"Nodes without assigned position and will be omitted: {missing_nodes}")

        return pos


    
    def draw_graph(self, mode='normal'):

        """
        Draws the graph on the canvas, applying the display mode selected by the user.
        :param mode: Display mode (‘normal’ or ‘ai’).
        """

        self.ax.clear()
        self.ax.set_axis_off()


        if mode == 'normal':
            node_colors = [self.get_node_color(node, self.coloring_enabled, mode='normal') for node in self.pos.keys()]
        elif mode == 'ai':
            node_colors = [self.get_node_color(node, coloring_enabled=True, mode='ai') for node in self.pos.keys()]
        else:
            node_colors = ['gray' for _ in self.pos.keys()]


        node_sizes = [self.node_size for _ in self.pos.keys()]

        try:
            nx.draw_networkx_nodes(
                self.G, self.pos, nodelist=self.pos.keys(), node_size=node_sizes, node_color=node_colors,
                ax=self.ax, linewidths=0, edgecolors='none'
            )
        except KeyError as err:
            messagebox.showerror("Drawing Error", f"The node '{err.args[0]}' does not have an assigned position.")
            print(f"Drawing Error: {err}")
            return

        edges_to_draw = [(u, v) for u, v in self.G.edges() if u in self.pos and v in self.pos]
        missing_edges = [(u, v) for u, v in self.G.edges() if u not in self.pos or v not in self.pos]
        if missing_edges:
            print(f"Edges omitted due to lack of positions: {missing_edges}")

        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=edges_to_draw, arrowstyle='->', arrowsize=10, ax=self.ax
        )

        labels = {node: self.id_a_concept.get(node, node) for node in self.pos.keys()}
        nx.draw_networkx_labels(
            self.G, self.pos, labels=labels, font_size=self.font_size, font_family='sans-serif', ax=self.ax  
        )

        self.fig.tight_layout() 
        self.canvas.draw_idle()



    def get_node_color(self, node, coloring_enabled=True, mode='normal'):

        """
        Determines the colour of each node according to its category and display mode.
        :param node: node of the network.
        :param colouring_enabled: Indicates whether colouring is enabled.
        :param mode: Display mode (‘normal’ or ‘ai’).
        :return: Colour assigned to the node.
        """

        if mode == 'ai' and self.ai_classification and node in self.ai_classification:
            if self.ai_classification[node] == 'framework':
                return 'orange'
            elif self.ai_classification[node] == 'projection':
                return 'cyan'
            else:
                return 'gray'
        if not coloring_enabled or not self.active_colors:
            return 'gray'  
        if node == self.initial_node:
            return self.active_colors.get('initial_node', 'gold')  
        elif node in self.nodes_to_colour_red:
            return self.active_colors.get('red', 'red') 
        elif node in self.nodes_to_colour_green:
            return self.active_colors.get('green', 'green') 
        elif node in self.conceptual_quarks:
            return self.active_colors.get('quark', 'green')  
        else:
            return self.active_colors.get('default', 'skyblue')  


    def toggle_coloring(self):

        """
        Toggles the network layout mode between ‘tree’ and ‘radial’, and redraws the network.
        """

        self.coloring_enabled = not self.coloring_enabled
        if self.coloring_enabled:
            self.toggle_button.config(text="Deactivate Colours")
        else:
            self.toggle_button.config(text="Activate Colours")
        self.draw_graph()

    
    def toggle_layout(self):

        """
        Toggles the network layout mode between ‘tree’ and ‘radial’, and redraws the network.
        """

        if self.layout_mode == 'tree':
            self.layout_mode = 'radial'
        else:
            self.layout_mode = 'tree'
        self.pos = self.get_layout()
        self.draw_graph()

    def start_ai_analysis(self):

        """
        Start a scan of the nodes using ai in a separate thread to keep the interface running.
        """

        confirm = messagebox.askyesno("Confirmar", "¿Deseas iniciar el análisis empírico/projection con ai?")
        if confirm:
            threading.Thread(target=self.ai_analysis, daemon=True).start()

    def ai_analysis(self):

        """
        It analyses each concept using the OpenAI API to classify it as a ‘framework’ or a ‘projection’, i.e. as conceptual or as empirical.
        """

        OPENAI_API_KEY = ""  
        if not OPENAI_API_KEY:
            messagebox.showerror("Error", "The OpenAI API key is not configured.")
            return
        openai.api_key = OPENAI_API_KEY

        self.ai_classification = {}  

        total_nodes = len(self.G.nodes())
        processed = 0

        for node in self.G.nodes():
            concept = self.id_a_concept.get(node, node)
            clasification = self.classify_concept(concept)
            if clasification and 'classification' in clasification:
                self.ai_classification[node] = clasification['classification']
                print(f"concept: {concept}, Clasification: {clasification['classification']}")  
            else:
                self.ai_classification[node] = 'unknown'
                print(f"concept: {concept}, Clasification: unknown")  
            processed += 1
            self.update_progress(processed, total_nodes)

        messagebox.showinfo("Success", "Empirical analysis/projection completed.")
        self.draw_graph(mode='ai')

    def classify_concept(self, concept):

        """
        Classify a concept as a ‘framework’ or ‘projection’ using ai with structured JSON output.
        :param concept: The concept to classify.
        :return: Dictionary with the classification.
        """

        concept = concept.lower()  

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence that classifies concepts into two categories: ‘framework’ or ‘projection’. "
                    "If the concept is real and refers to anything used to measure, quantities, actions, objects, instances, or concepts related to the real world, classify it as a ‘framework’. "
                    "If the concept seems abstract and 100% not directly related to reality, classify it as a ‘projection’. "
                    "It responds only with a JSON object structured in the following format: {\"classification\": \"framework\"} o {\"classification\": \"projection\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Classify the following concept: ‘{concept}’. "
                    f"Respond only with a JSON object in the format: {{\"classification\": \"framework\"}} o {{\"classification\": \"projection\"}}."
                ),
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "description": "Classification of the concept as ‘framework’ or ‘projection’.",
                            "type": "string"
                        }
                    },
                    "required": ["classification"],
                    "additionalProperties": False
                }
            }
        }

        try:
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages,
                response_format=response_format
            )
            contenido_respuesta = response.choices[0].message.content.strip()
            try:
                json_response = json.loads(contenido_respuesta)
                clasification = json_response.get("classification", "unknown").lower()
                if clasification in ['framework', 'projection']:
                    return {"classification": clasification}
                else:
                    return {"classification": "unknown"}
            except json.JSONDecodeError:
                print(f"Error parsing JSON response for concept '{concept}': {contenido_respuesta}")
                return {"classification": "unknown"}
        except Exception as e:
            print(f"Error when processing the Clasification with OpenAI for the concept '{concept}': {str(e)}")
            return {"classification": "unknown"}
        
    def update_progress(self, processed, total):

        """
        Updates the window with the progress in % of the analysis.
        :param processed: Number of processed nodes.
        :param total: Total number of nodes.
        """

        progress = (processed / total) * 100
        self.title(f"Conceptual Deconstruction Visualisation - Analysis: {progress:.2f}% completed")
        self.update_idletasks()

    def reset_ai_coloring(self):
        
        """
        Revert to the original colours by deactivating ai-based colouring or grey colouring.
        """

        if not self.ai_classification:
            messagebox.showinfo("Information", "No empirical analysis has been carried out.")
            return
        confirm = messagebox.askyesno("Confirm", "Would you like to return to the original colours?")
        if confirm:
            self.draw_graph(mode='normal')
            self.title("Conceptual Deconstruction Visualisation")

    def on_scroll(self, event):

        """
        Allows zooming in the network.
        :param event: Scroll event.
        """

        base_scale = 1.1
        if event.button == 'up':
            scale_factor = base_scale
        elif event.button == 'down':
            scale_factor = 1 / base_scale
        else:
            scale_factor = 1

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        new_width = (current_xlim[1] - current_xlim[0]) * scale_factor
        new_height = (current_ylim[1] - current_ylim[0]) * scale_factor

        relx = (current_xlim[1] - xdata) / (current_xlim[1] - current_xlim[0])
        rely = (current_ylim[1] - ydata) / (current_ylim[1] - current_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()



def main():

    """
    Main function that executes the concept deconstruction and launches the graphical application.
    If you want to use the UI, this is where you must specify initial parameters such as :initial_concept:, :max_depth: and :max_subdivisions:
    """

    initial_concept = "Mathematics".lower()
    max_depth = 3
    max_subdivisions = 2  

    deconstruction = ConceptualDeconstruction(initial_concept, max_depth, max_subdivisions)
    result = deconstruction.process()

    print("The result of Conceptual Deconstruction:")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    G = nx.DiGraph()

    numbered_concept = {}
    node_levels = {}
    id_a_concept = {}

    for concept_id in result.get('Analysed Concepts', []):
        concept_name = concept_id.split('_')[0].lower()
        suffix = concept_id.split('_')[1]
        concept_with_suffix = f"{concept_name}_{suffix}".lower()
        numbered_concept[concept_id] = concept_with_suffix
        id_a_concept[concept_with_suffix] = concept_name
        try:
            level = int(suffix.split('.')[0])
            node_levels[concept_with_suffix] = level
        except ValueError:
            print(f"Error when extracting suffix level '{suffix}' for the concept '{concept_id}'")
            node_levels[concept_with_suffix] = 0  

    initial_node = f"{initial_concept}_0.1.0"
    node_levels[initial_node] = 0

    if initial_node not in G:
        G.add_node(initial_node)

    concept_a_ids = {}
    for concept_id, concept_with_suffix in numbered_concept.items():
        concept_name = concept_id.split('_')[0].lower()
        if concept_name not in concept_a_ids:
            concept_a_ids[concept_name] = []
        concept_a_ids[concept_name].append(concept_id)

    numbered_branches = []
    for branch in result.get('Branch History', []):
        numbered_branch = []
        for concept in branch:
            concept = concept.lower()
            possible_ids = concept_a_ids.get(concept, [])
            if possible_ids:
                concept_id = min(possible_ids, key=lambda cid: node_levels.get(numbered_concept.get(cid), 0))
                concept_with_suffix = numbered_concept.get(concept_id, concept)
                numbered_branch.append(concept_with_suffix)
            else:
                if concept == initial_concept.lower():
                    numbered_branch.append(initial_node)
                else:
                    numbered_branch.append(concept)
        numbered_branches.append(numbered_branch)

    for numbered_branch in numbered_branches:
        for i in range(len(numbered_branch) - 1):
            parent = numbered_branch[i]
            child = numbered_branch[i + 1]
            if node_levels.get(child, 0) == node_levels.get(parent, 0) + 1:
                G.add_edge(parent, child)

    print("Network nodes:", G.nodes())
    print("Edges of the graph:", G.edges())

    nodes_to_colour_green = set()
    nodes_to_colour_red = set()

    def extract_name(concept):
        name = concept.split('_')[0]
        name = re.sub(r'\d+', '', name) 
        return name

    def are_similar_custom_gui(concept1, concept2, threshold=0.75):
        concept1 = concept1.lower()
        concept2 = concept2.lower()
        
        len1 = len(concept1)
        len2 = len(concept2)
        min_len = min(len1, len2)
        N = max(1, round(min_len * threshold))  

        sub_concept1 = concept1[:N]
        sub_concept2 = concept2[:N]

        return sub_concept1 == sub_concept2

    ancestors = {node: nx.ancestors(G, node) for node in G.nodes()}

    childless_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]

    for node in childless_nodes:
        node_name = extract_name(id_a_concept.get(node, node))
        for other_node in G.nodes():
            if other_node == node:
                continue
            other_name = extract_name(id_a_concept.get(other_node, other_node))
            if are_similar_custom_gui(node_name, other_name):
                if other_node in ancestors[node] or node in ancestors[other_node]:
                    nodes_to_colour_red.add(node)
                    nodes_to_colour_red.add(other_node)

    for node in childless_nodes:
        if node not in nodes_to_colour_red:
            nodes_to_colour_green.add(node)

    conceptual_quarks = nodes_to_colour_green.union(nodes_to_colour_red)

    print(f"Conceptual Quarks (Total: {len(conceptual_quarks)}): {conceptual_quarks}")

    app = TreeApp(
        G=G,
        initial_node=initial_node,
        node_levels=node_levels,
        id_a_concept=id_a_concept,
        conceptual_quarks=conceptual_quarks,
        nodes_to_colour_red=nodes_to_colour_red,
        nodes_to_colour_green=nodes_to_colour_green
    )
    app.mainloop()


if __name__ == "__main__":
    main()
