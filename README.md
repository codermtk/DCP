# Programmed Conceptual Deconstruction (PCD)

This repository contains the code necessary to apply the Programmed Conceptual Deconstruction (PCD) mentioned in the essay:  
*"Logic is Not Universal: Dismantling the Illusion of Reason"*.

Este repositorio contiene el código necesario para aplicar la Deconstrucción Conceptual Programada (DCP) mencionada en el ensayo:  
*"La Lógica No es Universal: Desmantelando la Ilusión de la Razón"*.

## Languages / Idiomas

- [English](#instructions-in-english)
- [Español](#instrucciones-en-español)

## Instructions in English 

### Requirements
1. Install Python 3.8 or higher
2. Install the required dependencies from requirements.txt:
--   pip install -r requirements.txt

### Usage
1. Set your OpenAI API key:
  Open the files DCP_en.py and UIDCP_en.py with your code editor.
  Locate the OPENAI_API_KEY variable at the beginning of the file and insert your OpenAI API key.
  Example:
  OPENAI_API_KEY = "your_api_key_here"
2. Choose the OpenAI model to use (gpt-4o or gpt-4o-mini both work well in general).
   Locate the model variable at the beginning of the file and select the model you would like to use.
   Example:
   model = "gpt-4o"
3. Set up the initial parameters.
   Open with your code editor the file UIDCP_en.py.
   1. Select the concept or field of knowledge you would like to apply PCD on.
   Example:
   initial_concept_ = "Mathematics".lower()
   2. Select the max depth (max number of subdivisions per branch) for your PCD.
   Example:
   max_depth_ = 5
   3. Select the max number of concepts that the PCD is allowed to deconstruct a concept into.
   Example:
   max_subdivisions_ = 4

## Instrucciones en Español

### Requisitos
1. Instalar Python 3.8 o superior
2. Instalar las dependencias requeridas desde requirements.txt:
--   pip install -r requirements.txt

### Uso
1. Configura tu clave API de OpenAI:
   Abre los archivos DCP_en.py y UIDCP_en.py con tu editor de código.
   Localiza la variable OPENAI_API_KEY al principio del archivo e inserta tu clave API de OpenAI.
   Ejemplo:
   OPENAI_API_KEY = "tu_api_key_aquí"
2. Elige el modelo de OpenAI a usar (gpt-4o o gpt-4o-mini ambos funcionan bien en general).
   Localiza la variable del modelo al principio del archivo y selecciona el modelo que deseas usar.
   Ejemplo:
   model = "gpt-4o"
3. Configura los parámetros iniciales.
   Abre con tu editor de código el archivo UIDCP_en.py.
   1. Selecciona el concepto o campo de conocimiento al que te gustaría aplicar la DCP.
   Ejemplo:
   initial_concept_ = "Mathematics".lower()
   2. Selecciona la profundidad máxima (número máximo de subdivisiones por rama) para tu DCP.
   Ejemplo:
   max_depth_ = 5
   3. Selecciona el número máximo de conceptos en los que la DCP puede deconstruir un concepto.
   Ejemplo:
   max_subdivisions_ = 4


   
