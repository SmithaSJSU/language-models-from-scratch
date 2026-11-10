"""Reproduce the 18 forward-equivalence checks; no training required."""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch import nn
from gru_scratch import GRUCellScratch, run_scratch_gru

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'rerun_results')
    args = parser.parse_args()
    rows = []
    for seed in [42, 1337, 2026]:
        for seq_len in [1, 7, 32]:
            for initial_state in ['zeros', 'random']:
                torch.manual_seed(seed)
                reference = nn.GRU(5, 8, batch_first=True).to(device='cpu', dtype=torch.float64)
                scratch = GRUCellScratch(5, 8).to(device='cpu', dtype=torch.float64)
                with torch.no_grad():
                    scratch.weight_ih.copy_(reference.weight_ih_l0)
                    scratch.weight_hh.copy_(reference.weight_hh_l0)
                    scratch.bias_ih.copy_(reference.bias_ih_l0)
                    scratch.bias_hh.copy_(reference.bias_hh_l0)
                x = torch.randn(3, seq_len, 5, dtype=torch.float64)
                h0 = (torch.zeros(1, 3, 8, dtype=torch.float64) if initial_state == 'zeros'
                      else torch.randn(1, 3, 8, dtype=torch.float64))
                with torch.no_grad():
                    output_ref, hidden_ref = reference(x, h0)
                    output_ours, hidden_ours = run_scratch_gru(scratch, x, h0)
                torch.testing.assert_close(output_ours, output_ref, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(hidden_ours, hidden_ref, atol=1e-12, rtol=1e-10)
                rows.append(dict(seed=seed, sequence_length=seq_len, initial_state=initial_state,
                    max_output_error=(output_ours-output_ref).abs().max().item(),
                    max_hidden_error=(hidden_ours-hidden_ref).abs().max().item(), passed=True))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/'gru_verification.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(torch_version=torch.__version__, device='cpu', dtype='float64',
        batch_size=3, input_size=5, hidden_size=8, num_layers=1, bidirectional=False,
        bias=True, atol=1e-12, rtol=1e-10, tests_passed=len(rows), total_tests=len(rows),
        worst_output_error=max(r['max_output_error'] for r in rows),
        worst_hidden_error=max(r['max_hidden_error'] for r in rows))
    (args.out_dir/'gru_verification_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
