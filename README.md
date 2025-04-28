# Preprocesamiento de matrices de conectividad fMRI

Este proyecto implementa un pipeline de preprocesamiento de datos de conectividad fMRI, aplicando transformaciones matemáticas específicas a cada canal y normalización estadística, preparando los tensores para tareas de machine learning.

---

## 📂 Estructura de carpetas esperada

```
GrandMeanNorm/
├── SubjectsDataAndTests.csv          # Metadata de sujetos
├── TensorData/                       # Tensores individuales por sujeto
│    └── {Grupo}_tensor_{SubjectID}.pt
├── config.yaml                       # Archivo de configuración
├── fold_1/
│    ├── train_original.pt
│    ├── train_post.pt
│    ├── train_z.pt
│    ├── val_original.pt
│    ├── val_post.pt
│    ├── val_z.pt
│    ├── test_original.pt
│    ├── test_post.pt
│    ├── test_z.pt
│    ├── train_labels.pt
│    ├── val_labels.pt
│    ├── test_labels.pt
│    └── train_stats.json
├── fold_2/
│    └── (idéntico a fold_1/)
├── fold_3/
├── fold_4/
├── fold_5/
└── (otros archivos de trabajo)
```

---

## 🛠️ Flujo de procesamiento

1. **Lectura de metadata**
   - `SubjectsDataAndTests.csv` contiene los IDs de sujetos, sexo, y grupo de investigación.

2. **Agrupamiento de tensores**
   - Se agrupan las rutas `.pt` por combinación `ResearchGroup + Sex` (`AD_M`, `CN_F`, etc.).

3. **Aplicación de transformaciones**
   - Para cada tensor de forma `(4, ROI, ROI)`, se aplican:
     - Fisher-Z a correlaciones.
     - Raíz cuadrada a MI y DCV.
     - Log(1+x) a Granger Causality.
   - **Importante**: no se aplica *clamping* previo. Se respeta el valor original del dato.

4. **Zero de diagonales**
   - Se anula la diagonal en los 4 canales para evitar sesgo.

5. **Split de datos**
   - Se realiza validación cruzada **estratificada** 5-fold (`StratifiedKFold`), manteniendo proporción entre grupos.

6. **Normalización z-score**
   - Se calcula `mean` y `std` en off-diagonales del training set.
   - Se aplica z-score a train, validation y test sets.

7. **Guardado**
   - Se guarda por fold:
     - Sets: `train_original.pt`, `train_post.pt`, `train_z.pt`, etc.
     - Etiquetas: `train_labels.pt`, `val_labels.pt`, `test_labels.pt`.
     - Estadísticas de normalización: `train_stats.json`.

8. **Visualización**
   - Se grafican:
     - Histogramas de distribución de off-diagonales (3 filas: original, post-transform, post+zscore).
     - Matrices de conectividad para un sujeto aleatorio (3 filas: original, post-transform, post+zscore).

---

## 📊 Notas sobre las transformaciones

| Canal            | Original                     | Transformación Aplicada | Rango Esperado |
|:-----------------|:------------------------------|:------------------------|:--------------|
| Correlación      | [-1, 1]                       | Fisher-Z                | ℝ (real line) |
| Mutual Information (MI) | [0, 1] (o más)         | Raíz cuadrada            | [0, 1] |
| Granger Causality | ≥ 0                         | log(1+x)                 | [0, ∞) |
| Dynamic Connectivity Variability (DCV) | [0, 1] (o más) | Raíz cuadrada         | [0, 1] |
