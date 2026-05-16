import sys
import json
from pathlib import Path

from lexetta_lcp.CompLexPerAnnotator import load_dataset, run_single_training, save_results, TrainingConfig


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <config.json> <run_name>")
    sys.exit(1)

config_path = Path(sys.argv[1])
run_name = sys.argv[2]

with open(config_path, "r") as f:
    config = TrainingConfig.model_validate_json(f.read())

output_dir = Path("outputs") / run_name
if output_dir.exists():
    print(f"Error: output directory already exists: {output_dir}")
    sys.exit(1)
output_dir.mkdir(parents=True)

with open(output_dir / "config.json", "w") as f:
    f.write(config.model_dump_json(indent=4))

data = load_dataset(val_size=config.val_split, test_size=config.test_split, seed=config.seed)

trainer, run = run_single_training(
    config=config,
    dataset=data,
    output_dir=str(output_dir / "checkpoints"),
)

print(f"Training time:    {run.metrics.train_time_s:.1f}s")
print(f"Peak VRAM:        {run.metrics.peak_vram_mb:.1f} MB")
print(f"Trainable params: {run.metrics.params_trainable:,} / {run.metrics.params_total:,}")
print(f"Train loss:       {run.metrics.final_train_loss:.4f}")
print(f"Eval loss:        {run.metrics.final_eval_loss:.4f}")

save_results(run, output_dir / "result.json")
trainer.save_model(str(output_dir / "model"))
print(f"Model saved to {output_dir / 'model'}")