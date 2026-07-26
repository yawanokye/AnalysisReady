from pathlib import Path
import pandas as pd

from statready.dispatch import run_analysis
from statready.reports import build_docx_report, build_excel_report, build_reproducibility_package

base = Path(__file__).resolve().parent
out = base / 'demo_output'
out.mkdir(exist_ok=True)

data = pd.read_csv(base / 'sample_data' / 'phase2_factor_sample.csv')
construct_map = {
    'ConstructA': ['a1', 'a2', 'a3', 'a4'],
    'ConstructB': ['b1', 'b2', 'b3', 'b4'],
}
paths = [('ConstructA', 'ConstructB')]
config = {
    'construct_map': construct_map,
    'paths': paths,
    'alpha': 0.05,
    'random_state': 42,
    'profile_variables': [],
}
result = run_analysis(data, 'sem', config)
study = {
    'title': 'StatReady Phase 2 Demonstration: Latent Construct Relationship',
    'objective': 'Estimate the latent effect of Construct A on Construct B after confirming their measurement structures.',
    'hypothesis': 'Construct A has a statistically significant positive effect on Construct B.',
    'alpha': 0.05,
    'method': 'Covariance-based structural equation model',
    'framework_notes': 'Construct A is specified as an exogenous latent predictor of Construct B.',
}
plan = pd.DataFrame([
    {'component': 'Objective', 'specification': study['objective']},
    {'component': 'Hypothesis', 'specification': study['hypothesis']},
    {'component': 'Method', 'specification': study['method']},
    {'component': 'Significance level', 'specification': 0.05},
    {'component': 'Construct measurement model', 'specification': 'ConstructA: a1, a2, a3, a4; ConstructB: b1, b2, b3, b4'},
    {'component': 'Structural paths', 'specification': 'ConstructA -> ConstructB'},
    {'component': 'Analysis dataset rows', 'specification': len(data)},
])

(out / 'StatReady_Phase2_Demo_Report.docx').write_bytes(build_docx_report(result, study, plan, []))
(out / 'StatReady_Phase2_Demo_Results.xlsx').write_bytes(build_excel_report(data, data.copy(), result, plan, []))
(out / 'StatReady_Phase2_Demo_Reproducibility.zip').write_bytes(build_reproducibility_package(data, data.copy(), result, study, plan, []))
for name, content in result.figures.items():
    safe = ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_')
    (out / f'{safe}.png').write_bytes(content)
print(result.summary)
print('Figures:', list(result.figures))
