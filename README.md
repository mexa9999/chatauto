# CitySim: нейроэволюция стратегий застройщиков

Мультиагентная модель города на NumPy: застройщики видят карту и принимают
решения о типе и качестве зданий, а нейроэволюция ищет стратегии, максимизирующие
прибыль при ограничениях на качество города и экстерналии.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Быстрый запуск

```powershell
.\.venv\Scripts\python.exe -m citysim demo --quick --policy manual --out runs/demo
.\.venv\Scripts\python.exe -m citysim train --quick --out runs/train_quick
.\.venv\Scripts\python.exe -m citysim experiment e1 --quick --out runs/e1_quick
```

## Структура

- `citysim/` — модель, политики, эволюция, эксперименты и визуализация;
- `tests/` — smoke- и unit-тесты;
- `docs/` — формальное описание модели и литература;
- `runs/` — сгенерированные артефакты (не коммитятся).

Подробности: `docs/MODEL.md`.
