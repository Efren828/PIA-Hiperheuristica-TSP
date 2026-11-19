# PIA — Hiper-heurística de Selección para el Problema del Agente Viajero (TSP)

**Autor:** Efrén Soto Martínez  
**Matrícula:** 2065612  
**Institución:** Facultad de Ciencias Físico Matemáticas (FCFM) — UANL  
**Carrera:** Licenciatura en Ciencias Computacionales  

## Descripción del Proyecto

Implementación de una **Hiper-heurística de Selección** basada en Aprendizaje Automático para predecir cuál de tres metaheurísticas/heurísticas de TSP —**Búsqueda Local (`2-opt hill climbing`)**, **Recocido Simulado (`SA + 2-opt`)** o **Aproximación con Árbol de Expansión Mínima (`MST + 2-opt`)**— obtiene el menor costo de recorrido sobre una instancia dada, utilizando características geométricas y estructurales del conjunto de ciudades.

## Estructura del Repositorio

```text
PIA-Hiperheuristica-TSP/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_labeled.csv
│   ├── test_labeled.csv
│   └── instances/
│       └── inst_0001.csv ... inst_0500.csv
├── src/
│   ├── generar_etiquetas.py
│   └── entrenar_evaluar_hh.py
└── results/
    ├── fig1_balance_clases.png
    ├── fig2_patrones_distribucion_tamano.png
    ├── fig3_matriz_confusion_modelos.png
    └── fig4_importancia_caracteristicas.png