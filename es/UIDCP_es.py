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

from DCP import DeconstruccionConceptual

OPENAI_API_KEY = "" 
client_openai = OpenAI(api_key=OPENAI_API_KEY)


def hierarchy_pos(G, root, niveles_nodos, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        
        """
        Esta función genera posiciones jerárquicas para los nodos

        :param G: El grafo dirigido (debe ser un árbol).
        :param root: Nodo raíz del árbol.
        :param niveles_nodos: Diccionario con los diferentes niveles de los nodos.
        :param width: Ancho del árbol.
        :param vert_gap: Espacio vertical entre niveles.
        :param vert_loc: Posición vertical del nodo raíz.
        :param xcenter: Posición horizontal central del nodo raíz.
        :return: El diccionario con las posiciones {nodo: (x, y)}.
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
        Cuenta el número de hojas descendientes de cualquier nodo dado.

        :param G: El grafo dirigido.
        :param node: Nodo desde el cual contar las hojas.
        :return: Número de hojas descendientes.
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
    Genera una disposición radial 360 grados para el grafo, y asigna diferentes radios según el nivel de profundidad del concepto.

    :param G: El grafo dirigido.
    :param root: Nodo raíz del grafo.
    :return: Diccionario de posiciones {nodo: (x, y)}.

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
    Es la clase principal de la aplicación,  crea la interfaz gráfica y visualiza el grafo resultante de la deconstrucción conceptual.
    """
    
    def __init__(self, G, nodo_inicial, niveles_nodos, id_a_concepto, quarks_conceptuales, nodos_a_colorear_rojo, nodos_a_colorear_verde):
        super().__init__()
        self.title("Visualización de Deconstrucción Conceptual")
        self.geometry("1600x900")
        self.coloring_enabled = True  
        self.active_colors = {
            'nodo_inicial': 'gold',
            'rojo': 'red',
            'verde': 'green',
            'quark': 'green',
            'default': 'skyblue'
        }
        self.layout_mode = 'tree'  
        self.ai_classification = {}  
        self.G = G
        self.nodo_inicial = nodo_inicial
        self.niveles_nodos = niveles_nodos
        self.id_a_concepto = id_a_concepto
        self.font_size = 10
        self.node_size = 700  
        self.quarks_conceptuales = quarks_conceptuales
        self.nodos_a_colorear_rojo = nodos_a_colorear_rojo  
        self.nodos_a_colorear_verde = nodos_a_colorear_verde  
        self.create_widgets()

    def create_widgets(self):

        """
        Crea los widgets de la interfaz gráfica, es decir, los botones y el área de dibujo para el grafo.
        """

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)


        layout_button = ttk.Button(control_frame, text="Cambiar Disposición", command=self.toggle_layout)
        layout_button.pack(side=tk.LEFT, padx=5)


        ia_button = ttk.Button(control_frame, text="Analizar Empírico/Conceptual", command=self.start_ai_analysis)
        ia_button.pack(side=tk.LEFT, padx=5)


        normal_colors_button = ttk.Button(control_frame, text="Volver a Colores Normales", command=self.reset_ai_coloring)
        normal_colors_button.pack(side=tk.LEFT, padx=5)


        reset_button = ttk.Button(control_frame, text="Analizar Concepto Nuevo", command=self.prompt_reset)
        reset_button.pack(side=tk.LEFT, padx=5)


        show_quarks_button = ttk.Button(control_frame, text="Mostrar Quarks Conceptuales", command=self.show_quarks)
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
        Muestra una lista con los Quarks Conceptuales encontrados y deja copiarlos y pegarlos.
        """

        if not self.quarks_conceptuales:
            messagebox.showinfo("Quarks Conceptuales", "No hay Quarks Conceptuales para mostrar.")
            return

        quarks_lista = [self.id_a_concepto.get(node, node) for node in self.quarks_conceptuales]
        quarks_texto = '\n'.join(quarks_lista)


        quarks_window = tk.Toplevel(self)
        quarks_window.title("Quarks Conceptuales")
        quarks_window.geometry("600x400")

        text_area = tk.Text(quarks_window, wrap=tk.WORD, font=("Arial", 10))
        text_area.insert(tk.END, quarks_texto)
        text_area.pack(expand=True, fill='both')


        text_area.config(state='normal')



    def prompt_reset(self):

        """
        Solicita al usuario nuevos parámetros y reinicia el proceso entero con el concepto o campo de conocimiento de elección del usuario.
        """

        try:
            concepto_inicial = simpledialog.askstring("Entrada", "Ingrese el concepto inicial:", parent=self)
            if concepto_inicial is None or concepto_inicial.strip() == "":
                messagebox.showerror("Error", "El concepto inicial no puede estar vacío.")
                return

            nivel_maximo = simpledialog.askinteger("Entrada", "Ingrese el nivel máximo de profundidad:", parent=self, minvalue=1)
            if nivel_maximo is None:
                messagebox.showerror("Error", "El nivel máximo es obligatorio.")
                return

            num_conceptos = simpledialog.askinteger("Entrada", "Ingrese el número de subdivisiones máximo por concepto:", parent=self, minvalue=1)
            if num_conceptos is None:
                messagebox.showerror("Error", "El número de conceptos es obligatorio.")
                return


            self.reset_graph(concepto_inicial, nivel_maximo, num_conceptos)

            self.pos = self.get_layout()
            self.draw_graph()
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al reiniciar el gráfico: {str(e)}")

    def reset_graph(self, concepto_inicial, nivel_maximo, num_conceptos):

        """
        Reinicia el gráfico con los nuevos parámetros .

        :param concepto_inicial: Concepto o campo de conocimiento raíz para la deconstrucción.
        :param nivel_maximo: Profundidad máxima de la deconstrucción, o número de subdivisiones máximo.
        :param num_conceptos: Número máximo de subdivisiones por concepto.
        """

        deconstruccion = DeconstruccionConceptual(concepto_inicial.lower(), nivel_maximo, num_conceptos)
        resultado = deconstruccion.procesar()

        print("Resultado de la Deconstrucción Conceptual:")
        print(json.dumps(resultado, indent=4, ensure_ascii=False))


        G = nx.DiGraph()


        concepto_numerado = {}
        niveles_nodos = {}
        id_a_concepto = {}


        for concepto_id in resultado.get('Conceptos Analizados', []):
            concepto_nombre = concepto_id.split('_')[0].lower()
            sufijo = concepto_id.split('_')[1]
            concepto_con_sufijo = f"{concepto_nombre}_{sufijo}".lower()
            concepto_numerado[concepto_id] = concepto_con_sufijo
            id_a_concepto[concepto_con_sufijo] = concepto_nombre

            nivel = int(sufijo.split('.')[0])
            niveles_nodos[concepto_con_sufijo] = nivel


        nodo_inicial = f"{concepto_inicial.lower()}_0.1.0"
        niveles_nodos[nodo_inicial] = 0


        if nodo_inicial not in G:
            G.add_node(nodo_inicial)


        concepto_a_ids = {}
        for concepto_id, concepto_con_sufijo in concepto_numerado.items():
            concepto_nombre = concepto_id.split('_')[0].lower()
            if concepto_nombre not in concepto_a_ids:
                concepto_a_ids[concepto_nombre] = []
            concepto_a_ids[concepto_nombre].append(concepto_id)


        ramas_numeradas = []
        for rama in resultado.get('Historial de Ramas', []):
            rama_numerada = []
            for concepto in rama:
                concepto = concepto.lower()

                ids_posibles = concepto_a_ids.get(concepto, [])
                if ids_posibles:

                    concepto_id = min(ids_posibles, key=lambda cid: niveles_nodos.get(concepto_numerado.get(cid), 0))
                    concepto_con_sufijo = concepto_numerado.get(concepto_id, concepto)
                    rama_numerada.append(concepto_con_sufijo)
                else:

                    if concepto == concepto_inicial.lower():
                        rama_numerada.append(nodo_inicial)
                    else:
                        rama_numerada.append(concepto)
            ramas_numeradas.append(rama_numerada)


        for rama_numerada in ramas_numeradas:
            for i in range(len(rama_numerada) - 1):
                parent = rama_numerada[i]
                child = rama_numerada[i + 1]

                if niveles_nodos.get(child, 0) == niveles_nodos.get(parent, 0) + 1:
                    G.add_edge(parent, child)

        print("Nodos del grafo:", G.nodes())
        print("Aristas del grafo:", G.edges())


        quarks_conceptuales = set()
        for quark in resultado.get('Quarks Conceptuales', []):
            if quark:

                quark_numerado = []
                for concepto in quark:
                    concepto = concepto.lower()
                    ids_posibles = concepto_a_ids.get(concepto, [])
                    if ids_posibles:

                        concepto_id = max(ids_posibles, key=lambda cid: niveles_nodos.get(concepto_numerado.get(cid), 0))
                        concepto_con_sufijo = concepto_numerado.get(concepto_id, concepto)
                        quark_numerado.append(concepto_con_sufijo)
                    else:
                        quark_numerado.append(concepto)

                quarks_conceptuales.add(quark_numerado[-1])


        bucles_conceptuales = set()
        for bucle in resultado.get('Bucles Conceptuales', []):
            if bucle:

                bucle_numerado = []
                for concepto in bucle:
                    concepto = concepto.lower()
                    ids_posibles = concepto_a_ids.get(concepto, [])
                    if ids_posibles:

                        for concepto_id in ids_posibles:
                            concepto_con_sufijo = concepto_numerado.get(concepto_id, concepto)
                            bucle_numerado.append(concepto_con_sufijo)
                    else:
                        bucle_numerado.append(concepto)

                if len(bucle_numerado) >= 2:
                    bucles_conceptuales.update(bucle_numerado)


        def son_similares_custom_gui(concepto1, concepto2, umbral=0.75):

            """
        Esta función comprueba si dos conceptos están escritos igual, de izquierda a derecha, y en un % que indica el umbral.

        :param concepto1: El primer concepto.
        :param concepto2: El segundo concepto.
        :param umbral: El % de coincidencia carácter por carácter que busca detectar la función, empezando por la izquierda.
        """
            
            concepto1 = concepto1.lower()
            concepto2 = concepto2.lower()
            

            len1 = len(concepto1)
            len2 = len(concepto2)
            min_len = min(len1, len2)
            N = max(1, round(min_len * umbral)) 


            sub_concepto1 = concepto1[:N]
            sub_concepto2 = concepto2[:N]

            return sub_concepto1 == sub_concepto2


        def extraer_nombre(concepto):
            
            nombre = concepto.split('_')[0]
            nombre = re.sub(r'\d+', '', nombre) 
            return nombre


        ancestros = {nodo: nx.ancestors(G, nodo) for nodo in G.nodes()}


        nodos_sin_hijos = [nodo for nodo in G.nodes() if G.out_degree(nodo) == 0]


        nodos_a_colorear_verde = set()
        nodos_a_colorear_rojo = set()

        for nodo in nodos_sin_hijos:
            nombre_nodo = extraer_nombre(id_a_concepto.get(nodo, nodo))
            for otro_nodo in G.nodes():
                if otro_nodo == nodo:
                    continue
                nombre_otro = extraer_nombre(id_a_concepto.get(otro_nodo, otro_nodo))
                if son_similares_custom_gui(nombre_nodo, nombre_otro):

                    if otro_nodo in ancestros[nodo] or nodo in ancestros[otro_nodo]:
                        nodos_a_colorear_rojo.add(nodo)
                        nodos_a_colorear_rojo.add(otro_nodo)


        for nodo in nodos_sin_hijos:
            if nodo not in nodos_a_colorear_rojo:
                nodos_a_colorear_verde.add(nodo)


        self.G = G
        self.nodo_inicial = nodo_inicial
        self.niveles_nodos = niveles_nodos
        self.id_a_concepto = id_a_concepto
        self.quarks_conceptuales = quarks_conceptuales
        self.bucles_conceptuales = bucles_conceptuales
        self.nodos_a_colorear_rojo = nodos_a_colorear_rojo  
        self.nodos_a_colorear_verde = nodos_a_colorear_verde 

    def get_layout(self):

        """
        Calcula las posiciones de todos los nodos del grafo según la disposición que se haya seleccionado.

        :return: Diccionario de posiciones {nodo: (x, y)}.
        """

        pos = {}

        reachable_nodes = set(nx.descendants(self.G, self.nodo_inicial)).union({self.nodo_inicial})
        

        subgraph_main = self.G.subgraph(reachable_nodes)
        
        if self.layout_mode == 'tree':

            sub_pos = hierarchy_pos(subgraph_main, self.nodo_inicial, self.niveles_nodos, width=30.0, vert_gap=3.0)
        elif self.layout_mode == 'radial':

            sub_pos = radial_layout(subgraph_main, self.nodo_inicial)
        else:

            sub_pos = nx.spring_layout(subgraph_main, seed=42)

        pos.update(sub_pos)


        missing_nodes = reachable_nodes - set(pos.keys())
        if missing_nodes:
            print(f"Nodos sin posición asignada y serán omitidos: {missing_nodes}")

        return pos


    
    def draw_graph(self, mode='normal'):

        """
        Dibuja el grafo en el canvas, aplicando el modo de visualización que el usuario seleccione.

        :param mode: Modo de visualización ('normal' o 'ia').
        """

        self.ax.clear()
        self.ax.set_axis_off()


        if mode == 'normal':
            node_colors = [self.get_node_color(node, self.coloring_enabled, mode='normal') for node in self.pos.keys()]
        elif mode == 'ia':
            node_colors = [self.get_node_color(node, coloring_enabled=True, mode='ia') for node in self.pos.keys()]
        else:
            node_colors = ['gray' for _ in self.pos.keys()]


        node_sizes = [self.node_size for _ in self.pos.keys()]

        try:
            nx.draw_networkx_nodes(
                self.G, self.pos, nodelist=self.pos.keys(), node_size=node_sizes, node_color=node_colors,
                ax=self.ax, linewidths=0, edgecolors='none'
            )
        except KeyError as err:
            messagebox.showerror("Error de Dibujo", f"El nodo '{err.args[0]}' no tiene una posición asignada.")
            print(f"Error de dibujo: {err}")
            return

        edges_to_draw = [(u, v) for u, v in self.G.edges() if u in self.pos and v in self.pos]
        missing_edges = [(u, v) for u, v in self.G.edges() if u not in self.pos or v not in self.pos]
        if missing_edges:
            print(f"Aristas omitidas por falta de posiciones: {missing_edges}")

        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=edges_to_draw, arrowstyle='->', arrowsize=10, ax=self.ax
        )

        labels = {node: self.id_a_concepto.get(node, node) for node in self.pos.keys()}
        nx.draw_networkx_labels(
            self.G, self.pos, labels=labels, font_size=self.font_size, font_family='sans-serif', ax=self.ax  
        )

        self.fig.tight_layout() 
        self.canvas.draw_idle()



    def get_node_color(self, node, coloring_enabled=True, mode='normal'):

        """
        Determina el color de cada nodo según su categoría y el modo de visualización.

        :param node: Nodo del grafo.
        :param coloring_enabled: Indica si el coloreo está habilitado.
        :param mode: Modo de visualización ('normal' o 'ia').
        :return: Color asignado al nodo.
        """

        if mode == 'ia' and self.ai_classification and node in self.ai_classification:
            if self.ai_classification[node] == 'marco':
                return 'orange'
            elif self.ai_classification[node] == 'proyeccion':
                return 'cyan'
            else:
                return 'gray'
        if not coloring_enabled or not self.active_colors:
            return 'gray'  
        if node == self.nodo_inicial:
            return self.active_colors.get('nodo_inicial', 'gold')  
        elif node in self.nodos_a_colorear_rojo:
            return self.active_colors.get('rojo', 'red') 
        elif node in self.nodos_a_colorear_verde:
            return self.active_colors.get('verde', 'green') 
        elif node in self.quarks_conceptuales:
            return self.active_colors.get('quark', 'green')  
        else:
            return self.active_colors.get('default', 'skyblue')  


    def toggle_coloring(self):

        """
        Alterna el modo de disposición del grafo entre 'tree' y 'radial', y redibuja el grafo.
        """

        self.coloring_enabled = not self.coloring_enabled
        if self.coloring_enabled:
            self.toggle_button.config(text="Desactivar Colores")
        else:
            self.toggle_button.config(text="Activar Colores")
        self.draw_graph()

    def select_colors(self):

        """
        Permite al usuario seleccionar colores personalizados para diferentes categorías de nodos.
        """

        color = colorchooser.askcolor(title="Seleccionar color para el nodo inicial")[1]
        if color:
            self.active_colors['nodo_inicial'] = color

        color = colorchooser.askcolor(title="Seleccionar color para nodos rojos")[1]
        if color:
            self.active_colors['rojo'] = color

        color = colorchooser.askcolor(title="Seleccionar color para nodos verdes")[1]
        if color:
            self.active_colors['verde'] = color

        color = colorchooser.askcolor(title="Seleccionar color para Quarks Conceptuales")[1]
        if color:
            self.active_colors['quark'] = color

        color = colorchooser.askcolor(title="Seleccionar color predeterminado")[1]
        if color:
            self.active_colors['default'] = color

        self.draw_graph()

    def toggle_layout(self):

        """
        Alterna el modo de disposición del grafo entre 'tree' y 'radial', y redibuja el grafo.
        """

        if self.layout_mode == 'tree':
            self.layout_mode = 'radial'
        else:
            self.layout_mode = 'tree'
        self.pos = self.get_layout()
        self.draw_graph()

    def start_ai_analysis(self):

        """
        Inicia un análisis de los nodos utilizando IA en un hilo separado para mantener la interfaz funcionando.
        """

        confirm = messagebox.askyesno("Confirmar", "¿Deseas iniciar el análisis empírico/proyeccion con IA?")
        if confirm:
            threading.Thread(target=self.ai_analysis, daemon=True).start()

    def ai_analysis(self):

        """
        Analiza cada concepto usando la API de OpenAI para clasificarlo como 'marco' o 'proyeccion', es decir, como conceptual o como empírico.
        """

        OPENAI_API_KEY = "sk-proj z6SsfvLwP5sk2s5qw_f1zTLAPj26iXZEeyiW4lh_EGOxrJHBj2PxreZdJiFAm6_kgFVOM83f_0T3BlbkFJTZkstbCgA6QZbDKCg5foK4cPP9hZvAnmxIcXe4a90lOneQXPXdsr_2wnetXEWXrwdcx5v1AQEA"  # Reemplaza con tu clave de API
        if not OPENAI_API_KEY:
            messagebox.showerror("Error", "La clave de API de OpenAI no está configurada.")
            return
        openai.api_key = OPENAI_API_KEY

        self.ai_classification = {}  

        total_nodes = len(self.G.nodes())
        processed = 0

        for node in self.G.nodes():
            concepto = self.id_a_concepto.get(node, node)
            clasificacion = self.classify_concept(concepto)
            if clasificacion and 'classification' in clasificacion:
                self.ai_classification[node] = clasificacion['classification']
                print(f"Concepto: {concepto}, Clasificación: {clasificacion['classification']}")  
            else:
                self.ai_classification[node] = 'desconocido'
                print(f"Concepto: {concepto}, Clasificación: desconocido")  
            processed += 1
            self.update_progress(processed, total_nodes)

        messagebox.showinfo("Éxito", "Análisis empírico/proyeccion completado.")
        self.draw_graph(mode='ia')

    def classify_concept(self, concepto):

        """
        Clasifica un concepto como 'marco' o 'proyeccion' usando IA con salida estructurada JSON.

        :param concepto: El concepto a clasificar.
        :return: Diccionario con la clasificación.
        """

        concepto = concepto.lower()  

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres una inteligencia artificial que clasifica conceptos en dos categorías: 'marco' o 'proyeccion'. "
                    "Si el concepto es real y se refiere a cualquier cosa usada para medir, cantidades, acciones, objetos, instancias, o conceptos relacionado con el mundo real clasifícalo como 'marco'. "
                    "Si el concepto te parece abstracto y 100% no está directamente relacionado con la realidad, clasifícalo como 'proyeccion'. "
                    "Responde solo con un objeto JSON estructurado en el siguiente formato: {\"classification\": \"marco\"} o {\"classification\": \"proyeccion\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Clasifica el siguiente concepto: '{concepto}'. "
                    f"Responde solo con un objeto JSON en el formato: {{\"classification\": \"marco\"}} o {{\"classification\": \"proyeccion\"}}."
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
                            "description": "Clasificación del concepto como 'marco' o 'proyeccion'.",
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
                respuesta_json = json.loads(contenido_respuesta)
                clasificacion = respuesta_json.get("classification", "desconocido").lower()
                if clasificacion in ['marco', 'proyeccion']:
                    return {"classification": clasificacion}
                else:
                    return {"classification": "desconocido"}
            except json.JSONDecodeError:
                print(f"Error al parsear la respuesta JSON para el concepto '{concepto}': {contenido_respuesta}")
                return {"classification": "desconocido"}
        except Exception as e:
            print(f"Error al procesar la clasificación con OpenAI para el concepto '{concepto}': {str(e)}")
            return {"classification": "desconocido"}
        
    def update_progress(self, processed, total):

        """
        Actualiza la ventana con el progreso en % del análisis.

        :param processed: Número de nodos procesados.
        :param total: Número total de nodos.
        """

        progress = (processed / total) * 100
        self.title(f"Visualización de Deconstrucción Conceptual - Análisis IA: {progress:.2f}% completado")
        self.update_idletasks()

    def reset_ai_coloring(self):
        
        """
        Vuelve a los colores originales desactivando la coloración basada en IA o la coloración gris.
        """

        if not self.ai_classification:
            messagebox.showinfo("Información", "No se ha realizado ningún análisis empírico/proyeccion.")
            return
        confirm = messagebox.askyesno("Confirmar", "¿Deseas volver a los colores originales?")
        if confirm:
            self.draw_graph(mode='normal')
            self.title("Visualización de Deconstrucción Conceptual")

    def on_scroll(self, event):

        """
        Permite hacer zoom en el grafo.

        :param event: Evento de scroll.
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
    Función principal que ejecuta la deconstrucción conceptual y lanza la aplicación gráfica.
    Si se quiere usar la UI, es aquí donde se deben de especificar los parámetros iniciales como :concepto_inicial:, :nivel_maximo: y :num_conceptos:
    """

    concepto_inicial = "Verdad".lower()
    nivel_maximo = 3
    num_conceptos = 2  

    deconstruccion = DeconstruccionConceptual(concepto_inicial, nivel_maximo, num_conceptos)
    resultado = deconstruccion.procesar()

    print("Resultado de la Deconstrucción Conceptual:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))

    G = nx.DiGraph()

    concepto_numerado = {}
    niveles_nodos = {}
    id_a_concepto = {}

    for concepto_id in resultado.get('Conceptos Analizados', []):
        concepto_nombre = concepto_id.split('_')[0].lower()
        sufijo = concepto_id.split('_')[1]
        concepto_con_sufijo = f"{concepto_nombre}_{sufijo}".lower()
        concepto_numerado[concepto_id] = concepto_con_sufijo
        id_a_concepto[concepto_con_sufijo] = concepto_nombre
        try:
            nivel = int(sufijo.split('.')[0])
            niveles_nodos[concepto_con_sufijo] = nivel
        except ValueError:
            print(f"Error al extraer el nivel del sufijo '{sufijo}' para el concepto '{concepto_id}'")
            niveles_nodos[concepto_con_sufijo] = 0  

    nodo_inicial = f"{concepto_inicial}_0.1.0"
    niveles_nodos[nodo_inicial] = 0

    if nodo_inicial not in G:
        G.add_node(nodo_inicial)

    concepto_a_ids = {}
    for concepto_id, concepto_con_sufijo in concepto_numerado.items():
        concepto_nombre = concepto_id.split('_')[0].lower()
        if concepto_nombre not in concepto_a_ids:
            concepto_a_ids[concepto_nombre] = []
        concepto_a_ids[concepto_nombre].append(concepto_id)

    ramas_numeradas = []
    for rama in resultado.get('Historial de Ramas', []):
        rama_numerada = []
        for concepto in rama:
            concepto = concepto.lower()
            ids_posibles = concepto_a_ids.get(concepto, [])
            if ids_posibles:
                concepto_id = min(ids_posibles, key=lambda cid: niveles_nodos.get(concepto_numerado.get(cid), 0))
                concepto_con_sufijo = concepto_numerado.get(concepto_id, concepto)
                rama_numerada.append(concepto_con_sufijo)
            else:
                if concepto == concepto_inicial.lower():
                    rama_numerada.append(nodo_inicial)
                else:
                    rama_numerada.append(concepto)
        ramas_numeradas.append(rama_numerada)

    for rama_numerada in ramas_numeradas:
        for i in range(len(rama_numerada) - 1):
            parent = rama_numerada[i]
            child = rama_numerada[i + 1]
            if niveles_nodos.get(child, 0) == niveles_nodos.get(parent, 0) + 1:
                G.add_edge(parent, child)

    print("Nodos del grafo:", G.nodes())
    print("Aristas del grafo:", G.edges())

    nodos_a_colorear_verde = set()
    nodos_a_colorear_rojo = set()

    def extraer_nombre(concepto):
        nombre = concepto.split('_')[0]
        nombre = re.sub(r'\d+', '', nombre) 
        return nombre

    def son_similares_custom_gui(concepto1, concepto2, umbral=0.75):
        concepto1 = concepto1.lower()
        concepto2 = concepto2.lower()
        
        len1 = len(concepto1)
        len2 = len(concepto2)
        min_len = min(len1, len2)
        N = max(1, round(min_len * umbral))  

        sub_concepto1 = concepto1[:N]
        sub_concepto2 = concepto2[:N]

        return sub_concepto1 == sub_concepto2

    ancestros = {nodo: nx.ancestors(G, nodo) for nodo in G.nodes()}

    nodos_sin_hijos = [nodo for nodo in G.nodes() if G.out_degree(nodo) == 0]

    for nodo in nodos_sin_hijos:
        nombre_nodo = extraer_nombre(id_a_concepto.get(nodo, nodo))
        for otro_nodo in G.nodes():
            if otro_nodo == nodo:
                continue
            nombre_otro = extraer_nombre(id_a_concepto.get(otro_nodo, otro_nodo))
            if son_similares_custom_gui(nombre_nodo, nombre_otro):
                if otro_nodo in ancestros[nodo] or nodo in ancestros[otro_nodo]:
                    nodos_a_colorear_rojo.add(nodo)
                    nodos_a_colorear_rojo.add(otro_nodo)

    for nodo in nodos_sin_hijos:
        if nodo not in nodos_a_colorear_rojo:
            nodos_a_colorear_verde.add(nodo)

    quarks_conceptuales = nodos_a_colorear_verde.union(nodos_a_colorear_rojo)

    print(f"Quarks Conceptuales (Total: {len(quarks_conceptuales)}): {quarks_conceptuales}")

    app = TreeApp(
        G=G,
        nodo_inicial=nodo_inicial,
        niveles_nodos=niveles_nodos,
        id_a_concepto=id_a_concepto,
        quarks_conceptuales=quarks_conceptuales,
        nodos_a_colorear_rojo=nodos_a_colorear_rojo,
        nodos_a_colorear_verde=nodos_a_colorear_verde
    )
    app.mainloop()


if __name__ == "__main__":
    main()
