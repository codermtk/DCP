import re
from openai import OpenAI
import json
import inflect


OPENAI_API_KEY = "" 
client_openai = OpenAI(api_key=OPENAI_API_KEY)
modelo_ = "gpt-4o-mini"
class DeconstruccionConceptual:
    """
   Clase para aplicar DCP a un concepto o campo de conocimiento.
    """
    def __init__(self, concepto_inicial, nivel_maximo, num_conceptos):

        """
        Inicializa la clase .

        :param concepto_inicial: El concepto o campo de conocimiento a descomponer.
        :param nivel_maximo: La profundidad máximo o número de subdivisiones máximo que aplicará el algoritmo.
        :param num_conceptos: El número máximo de subconceptos o subcampos que el algoritmo puede extraer por concepto o campo de conocimiento.
        """

        self.concepto_inicial = concepto_inicial.lower()
        self.nivel_maximo = nivel_maximo
        self.num_conceptos = num_conceptos
        self.lista_pendientes = [f"{self.concepto_inicial}_0.1.0"]
        self.lista_analizados = []
        self.waitlist = []
        self.p = inflect.engine()
        self.no_return = []
        self.quarks_conceptuales = []
        self.bucles_conceptuales = []
        self.ramas_conceptos = {self.concepto_inicial: []}
        self.historial_ramas = []

    def normalizar_concepto(self, concepto):

        """
        Pasa a un formato más fácil de procesar por el algoritmo el concepto o campo de conocimiento, y lo hace pasándolo a minúsculas todo.
        """

        return concepto.lower()

    def son_similares_custom(self, concepto1, concepto2, umbral=0.75):

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

    def extraer_bucle(self, rama, concepto_repetido):

        """
        Esta función extrae bucles conceptuales de ramas.

        :param rama: La rama sobre la que se busca extraer un bucle conceptual.
        :param concepto_repetido: El nombre del concepto o campo de conocimiento que se repite.
        """

        indices = [i for i, c in enumerate(rama) if c == concepto_repetido]
        if len(indices) >= 2:
            inicio = indices[0]
            fin = indices[-1]
            return rama[inicio:fin + 1]
        return []



    def descomponer_concepto(self, concepto, nivel, contexto):

        """
        Esta función descompone un concepto o campo de conocimiento en n número máximo de :param num_conceptos: conceptos o subcampos de conocimiento.
        Lo hace usando la API de OpenAI y el modelo que el usuario seleccione.
          

        :param concepto: El concepto a descomponer.
        :param nivel: El nivel de profundidad al que se encuentra el concepto a descomponer.
        :param contexto: El contexto son todos los conceptos de los que proviene el concepto a descomponer, es decir, la rama.
        """

        concepto = concepto.lower()  
       
        print(f"Descomponiendo concepto: '{concepto}', Nivel: {nivel}, Contexto: '{contexto}'")

        messages = [
            {
                "role": "system",
                "content": (
                    f"Eres un experto en definiciones."
                    f"Se te presentará un hilo de subdivisiones y tu trabajo es determinar los siguientes dos criterios:"
                    f"1.Si la cadena sigue un órden decreciente claro (Ejemplo hecho bien: Antigua Grecia -> Filosofía Griega -> Platón) (Ejemplo hecho mal: verdad -> hecho -> evidencia -> observación -> análisis)"
                    f"2.Si todos los términos de la cadena están relacionados por definición exacta."
                    f"Si no se cumple uno de esos dos criterios para el último término añadido a la cadena debes devolver en el formato JSON la palabra 'vacío' para todos los valores"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"El último término añadido a la cadena es: '{concepto}'. "
                    f"La cadena de subdivisiones que ha llevado a este término es: '{contexto}'."
                    f"Por favor, analiza si el término se puede subdividir en otros {self.num_conceptos} términos."                  
                    f"El formato debe ser: {{'concept_1': 'subdivision1', 'concept_2': 'subdivision2', ...}} "
                    f"Puedes devolver menos términos que los {self.num_conceptos}, devuelve 'vacío' en los sobrantes."
                    f"Te he estado probando y nunca pones 'vacío', recuerda poner 'vacío' en todo si ves un término en la cadena no se relaciona al 100% con el original"  
                    f"Si ves la cadena vacía es porque es el primer término, trata de no devolver vacío en dicho caso."
                    f"No tengas miedo a devolver todo 'vacío' si no puedes encontrar más términos relevantes a la cadena"
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
                            "description": f"El sustantivo {i + 1} más relevante extraído de la definición.",
                            "type": "string"
                        } for i in range(self.num_conceptos)
                    },
                    "required": [f"concept_{i + 1}" for i in range(min(self.num_conceptos, 1))],
                    "additionalProperties": False
                }
            }
        }

        try:

            response = client_openai.chat.completions.create(
                model=modelo_,  
                messages=messages,
                response_format=response_format

            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:

            return f"Error al procesar la definición con OpenAI: {str(e)}"

    
    def procesar(self):

        """
        Esta función es la encargada de la implementación de DCP y de coordinar el resto de funciones en este archivo.
        """

        while self.lista_pendientes:

            concepto_actual = self.lista_pendientes.pop(0)
            concepto_nombre, resto = concepto_actual.rsplit('_', 1)
            nivel, j, k = [int(x) for x in resto.split('.')]
            concepto_nombre = self.normalizar_concepto(concepto_nombre)

            rama_actual = self.ramas_conceptos.get(concepto_actual, [])

            contexto = " -> ".join(rama_actual)

            bucle_encontrado = False
            for ancestro in rama_actual:

                if self.son_similares_custom(concepto_nombre, ancestro):
                    bucle = self.extraer_bucle(rama_actual, ancestro) + [concepto_nombre]
                    self.bucles_conceptuales.append(bucle)
                    bucle_encontrado = True
                    break  

            if bucle_encontrado:

                continue

            subconceptos = self.descomponer_concepto(concepto_nombre, nivel, contexto)
            print(f"Subconceptos recibidos para '{concepto_nombre}': {subconceptos}")

            if 'Error' in subconceptos:

                self.no_return.append(concepto_actual)
                self.quarks_conceptuales.append(rama_actual + [concepto_nombre])
                continue

            nueva_rama = rama_actual + [concepto_nombre]
            try:
                subconceptos_dict = json.loads(subconceptos.replace("'", '"'))
            except json.JSONDecodeError as e:
                print(f"Error al parsear el JSON: {e}")
                print(f"Respuesta de la API: {subconceptos}")
                self.quarks_conceptuales.append(nueva_rama)
                continue

            if not subconceptos_dict:
                self.quarks_conceptuales.append(nueva_rama)
                continue

            tiene_subconceptos = False
            for idx, subconcepto in enumerate(subconceptos_dict.values(), start=1):
                subconcepto = subconcepto.lower()
                subconcepto = self.normalizar_concepto(subconcepto)
                if subconcepto == 'vacío':
                    continue  
                else:
                    subconcepto_id = f"{subconcepto}_{nivel + 1}.{idx}.{j}"
                    similar_a_ancestro = False
                    ancestro_repetido = None  
                    for ancestro in nueva_rama:
                        if self.son_similares_custom(subconcepto, ancestro):
                            similar_a_ancestro = True
                            ancestro_repetido = ancestro
                            break
                    if similar_a_ancestro:
                        bucle = self.extraer_bucle(nueva_rama, ancestro_repetido) + [subconcepto]
                        self.bucles_conceptuales.append(bucle)
                        continue
                    else:
                        tiene_subconceptos = True
                        if nivel + 1 <= self.nivel_maximo:
                            self.lista_pendientes.append(subconcepto_id)
                        self.ramas_conceptos[subconcepto_id] = nueva_rama
                        self.historial_ramas.append(nueva_rama + [subconcepto])

            if not tiene_subconceptos:
                self.quarks_conceptuales.append(nueva_rama)

            self.lista_analizados.append(concepto_actual)

        bucles_aplanados = set()
        for bucle in self.bucles_conceptuales:
            bucles_aplanados.update(bucle)

        return {

            "Conceptos Analizados": self.lista_analizados,
            "Quarks Conceptuales": self.quarks_conceptuales,
            "Bucles Conceptuales": list(self.bucles_conceptuales),
            "Bucles Aplanados": list(bucles_aplanados),
            "Historial de Ramas": self.historial_ramas

        }

    
def main():
    """
    Función principal del script.
    Si se empleará únicamente el algoritmo sin la UI, se deben de indicar los parámetros iniciales aquí.
    Desde el concepto o campo de conocimiento a deconstruir :concepto_inicial:, al número de subdivisiones máximo :nivel_maximo: 
    al número de subconceptos o subcampos de conocimiento máximo en los que puede deconstruir el algoritmo cada concepto o campo de conocimiento :num_conceptos:.
    Si se pretende usar la UI, estos parámetros se configuran en UIDCP.py, no aquí.
    """

    concepto_inicial = ""
    nivel_maximo = 6
    num_conceptos = 5
    deconstruccion = DeconstruccionConceptual(concepto_inicial, nivel_maximo, num_conceptos)
    resultado = deconstruccion.procesar()
    print("Resultado de la Deconstrucción Conceptual:")
    print(resultado)

if __name__ == "__main__":
    main()