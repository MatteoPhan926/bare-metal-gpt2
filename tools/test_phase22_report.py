import unittest
from phase22_report import aggregate


class AggregateTests(unittest.TestCase):
    def test_all_processes_and_spread_retained(self):
        data=[]
        for median in (9.,1.,4.):
            row={'n':256}
            for metric in ('event_ms','wall_ms','enqueue_ms','update_ms'):
                row[metric]={'median':median,'min':0.,'max':100.,'q1':median-.1,'q3':median+.1}
            data.append({('generation','ordered4',1007):row})
        result=aggregate(data)[0]
        self.assertEqual(result['n_each'],[256,256,256])
        self.assertEqual(result['wall_ms']['median_of_medians'],4.)
        self.assertEqual(result['wall_ms']['run_medians'],[9.,1.,4.])
        self.assertEqual(result['wall_ms']['all_sample_max'],100.)
        self.assertEqual(len(result['wall_ms']['run_iqrs']),3)

    def test_missing_condition_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([{('generation','original',128):{}},{}])


if __name__=='__main__': unittest.main()
