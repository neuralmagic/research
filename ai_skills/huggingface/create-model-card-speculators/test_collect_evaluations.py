import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from collect_evaluations import collect_aiperf_artifacts


class CollectAIPerfArtifactsTest(unittest.TestCase):
    def test_collects_known_tables_plots_commands_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "acceptance.csv").write_text(
                "dataset,acceptance_length,acceptance_rate_pct\n"
                "humaneval,3.5,62.0\n",
                encoding="utf-8",
            )
            (root / "acceptance_by_prompt_length.csv").write_text(
                "dataset,prompt_length,requests,acceptance_length\n"
                "mrcr,8193-16384,4,3.2\n",
                encoding="utf-8",
            )
            (root / "acceptance_length_by_category.csv").write_text(
                "metric,coding,math,Overall\naccept_length,4.1,3.9,4.0\n",
                encoding="utf-8",
            )
            (root / "acceptance_rate_by_category.csv").write_text(
                "metric,coding,math,Overall\naccept_rate,0.71,0.66,0.685\n",
                encoding="utf-8",
            )
            (root / "eval_command.txt").write_text(
                "python scripts/evaluate/evaluate.py acceptance --dataset humaneval\n",
                encoding="utf-8",
            )
            (root / "plots").mkdir()
            (root / "plots" / "humaneval.png").write_bytes(b"png")
            (root / "profile_export_aiperf.json").write_text(
                "{}\n", encoding="utf-8"
            )

            result = collect_aiperf_artifacts(root)

            self.assertTrue(result["found_anything"])
            self.assertEqual(result["modes"], ["acceptance"])
            self.assertEqual(
                result["tables"]["acceptance.csv"][0]["rows"][0]["dataset"],
                "humaneval",
            )
            self.assertEqual(
                result["tables"]["acceptance_rate_by_category.csv"][0]["rows"][0][
                    "Overall"
                ],
                "0.685",
            )
            self.assertEqual(result["plots"], ["plots/humaneval.png"])
            self.assertEqual(
                result["native_summaries"], ["profile_export_aiperf.json"]
            )
            self.assertIn("evaluate.py acceptance", result["commands"][0]["content"])

    def test_cli_combines_model_card_and_aiperf_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            evaluation = root / "evaluation"
            model.mkdir()
            evaluation.mkdir()
            (model / "README.md").write_text("# Draft model\n", encoding="utf-8")
            (evaluation / "acceptance.csv").write_text(
                "dataset,acceptance_length,acceptance_rate_pct\n"
                "humaneval,3.5,62.0\n",
                encoding="utf-8",
            )
            (evaluation / "eval_command.txt").write_text(
                "python scripts/evaluate/evaluate.py speedup --dataset humaneval\n",
                encoding="utf-8",
            )
            (evaluation / "profile_export_aiperf.json").write_text(
                "{}\n", encoding="utf-8"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("collect_evaluations.py")),
                    str(model),
                    "--eval-dir",
                    str(evaluation),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)

            self.assertEqual(result["style"], "aiperf")
            self.assertEqual(result["datasets"], ["humaneval"])
            self.assertEqual(result["aiperf_artifacts"][0]["modes"], ["speedup"])
            self.assertNotIn("acceptance_rates", result["missing"])
            self.assertNotIn("performance", result["missing"])


if __name__ == "__main__":
    unittest.main()
