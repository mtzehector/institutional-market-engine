# Instrucciones de integración

Este paquete contiene únicamente documentación fundacional. No modifica código Python.

## 1. Respaldo

```bat
git status
git pull origin main
```

## 2. Copiar archivos

Descomprime el ZIP sobre la raíz de `institutional-market-engine` y permite reemplazar:

- `README.md`
- `PROJECT_PHILOSOPHY.md`
- `RESEARCH_JOURNAL.md`

Los demás archivos son nuevos.

## 3. Revisar cambios

```bat
git status
git diff -- README.md PROJECT_PHILOSOPHY.md RESEARCH_JOURNAL.md
git diff --no-index NUL RESEARCH_MANIFESTO.md
git diff --no-index NUL RESEARCH_ROADMAP.md
```

## 4. Ejecutar pruebas

Aunque no hay cambios de producción:

```bat
pytest
```

## 5. Commit fundacional

```bat
git add README.md PROJECT_PHILOSOPHY.md RESEARCH_MANIFESTO.md RESEARCH_ROADMAP.md RESEARCH_JOURNAL.md DISCOVERIES.md BASELINES.md LEGACY.md CHANGELOG.md
git commit -m "docs(research): establish the scientific foundation of the Institutional Market Research Platform"
git push origin main
```

## 6. Tag opcional

```bat
git tag -a research-foundation-1.0 -m "The project formally establishes its scientific mission, vision and research philosophy"
git push origin research-foundation-1.0
```
