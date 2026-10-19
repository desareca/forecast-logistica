# Diagramas de arquitectura (Tarea 5)

| Archivo | Contenido |
|---|---|
| `01_arquitectura` | Componentes AWS: batch, persistencia, inferencia y observabilidad |
| `02_batch_diario` | Flujo del batch diario: calidad, gate, entrenamiento, backtest y publicación |
| `03_inferencia` | Secuencia de una inferencia, con fallback al pronóstico base |

Cada diagrama está como fuente Mermaid (`.mmd`) y exportado a `.svg` (para documentos) y `.png` (para el deck).

La fuente son los `.mmd`; los notebooks 00 y 06 muestran los `.png` (GitHub no renderiza Mermaid dentro de
notebooks). Si se edita un `.mmd`, hay que regenerar sus imágenes y volver a ejecutar esos notebooks:

```bash
npx -y @mermaid-js/mermaid-cli@11.4.0 -i 01_arquitectura.mmd -o 01_arquitectura.svg -b white
npx -y @mermaid-js/mermaid-cli@11.4.0 -i 01_arquitectura.mmd -o 01_arquitectura.png -b white -s 2
```
