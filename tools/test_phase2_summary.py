"""CPU-only checks for the statistical harness: python -m unittest discover -s tools -p test_phase2_summary.py"""
from pathlib import Path
import tempfile
import unittest
from phase2_summary import summarize

HEADER="sample,experiment,policy,ctx,pos,rep,token,event_ms,wall_ms,enqueue_ms,update_ms\n"

class SummaryTests(unittest.TestCase):
    def test_median_and_spread_not_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"stdout.txt").write_text("metadata,ignored\n"+HEADER+"".join(
                f"sample,fixed_batch16,ordinary,128,128,{i},1,{v},{v},0,0\n"
                for i,v in enumerate((8,2,10,4))))
            r=summarize(p)[0]
            self.assertEqual(r["n"],4) # four batch averages, never 64 samples
            self.assertEqual(r["event_ms"]["median"],6)
            self.assertEqual(r["event_ms"]["min"],2)
            self.assertEqual(r["event_ms"]["max"],10)
    def test_nonfinite_and_negative_fail(self):
        for value in ("nan","inf","-1"):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)
                (p/"stdout.txt").write_text(HEADER+f"sample,x,graph,128,128,0,1,{value},1,0,0\n")
                with self.assertRaises(ValueError):summarize(p)
    def test_missing_samples_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"stdout.txt").write_text("nothing\n")
            with self.assertRaises(ValueError):summarize(p)

if __name__=="__main__":unittest.main()
