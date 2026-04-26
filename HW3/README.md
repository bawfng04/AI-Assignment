# HW3 Topic 3 - Class Scheduling GA

## Muc tieu

Du an nay giai bai toan xep lich lop hoc bang Genetic Algorithm.
Moi run se tao:

- lich hoc cuoi cung
- file JSON ket qua
- file CSV ket qua
- bieu do fitness

## Chay nhanh nhat

1. Cai package:

```bash
pip install -r requirements.txt
```

2. Chay GA bang CLI:

```bash
python run.py --seed 42 --offerings 30
```

3. Chay UI demo:

```bash
streamlit run app.py
```

## Cac file chinh

- run.py: entrypoint CLI
- app.py: UI Streamlit de demo
- scripts/report.py: chay multi-seed va xuat bang tong hop
- topic3_ga/: toan bo code GA (config, data, fitness, operators, engine, export)
- tests/: unit test + smoke test
- report/methods.md: mo ta Fitness/Selection/Crossover/Mutation de dua vao bao cao

## Tham so cot loi (mac dinh)

- population_size = 240
- generations = 420
- crossover_rate = 0.95
- mutation_rate = 0.20
- elitism_count = 10
- tournament_size = 4
- hard_penalty_weight = 1000.0
- soft_penalty_weight = 1.0
- no_improvement_patience = 100
- feasible_streak_patience = 100

## Cac rang buoc hard

- offering co 2 session, khong cung ngay, khong ngay lien ke
- room size phai phu hop class size
- professor khong trung gio
- room phai available
- section khong hoc 2 phong cung luc
- 1 phong khong chua 2 section cung luc
- professor khong qua 3 courses

## Artifact output

Moi run tao thu muc:

- outputs/{run_name}_seed{seed}/

Trong do co:

- best_schedule.json
- best_schedule.csv
- fitness_plot.png

## Bao cao multi-seed

Lenh:

```bash
python scripts/report.py --seeds 40,41,42,43,44 --offerings 30 --run-name report_ga
```

File tong hop tao ra:

- outputs/report_ga_summary.csv
- outputs/report_ga_summary.md
- outputs/report_ga_summary_stats.json

## Benchmark giu lai cho bao cao

Dang giu san cac ket qua benchmark:

- outputs/report_ga_seed40/
- outputs/report_ga_seed41/
- outputs/report_ga_seed42/
- outputs/report_ga_seed43/
- outputs/report_ga_seed44/
- outputs/report_ga_summary.csv
- outputs/report_ga_summary.md
- outputs/report_ga_summary_stats.json

## Test

Chay toan bo test:

```bash
python -m unittest discover -s tests -v
```
