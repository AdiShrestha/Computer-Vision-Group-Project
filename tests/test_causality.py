import unittest
import numpy as np
from core.confseg_core import CausalController, FeedbackReplay, paired_calibration_draw

class SpyController(CausalController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.inputs_at_decision=[]
    def step(self,t,features=None):
        self.inputs_at_decision.append((t,tuple(self.received_feedback)))
        return super().step(t,features)

class Causality(unittest.TestCase):
    def test_feedback_boundary_first_timing_and_sparse_ledger(self):
        ctl=SpyController(.5,method='intercept_pid',latency=3,sparse=2)
        ev=FeedbackReplay(ctl)
        ev.run([.9,.8,.7,.6,.5,.4,.3,.2])
        self.assertEqual(ctl.inputs_at_decision[0][1],())
        self.assertEqual(ctl.inputs_at_decision[1][1],())
        self.assertEqual(ctl.inputs_at_decision[2][1],())
        self.assertEqual(ctl.inputs_at_decision[3][1],((3,0,.9),))
        self.assertEqual(ctl.inputs_at_decision[4][1],((3,0,.9),))
        self.assertEqual(ctl.inputs_at_decision[5][1],((3,0,.9),(5,2,.7)))
        self.assertEqual(ctl.inputs_at_decision[6][1][-1],(5,2,.7))
        self.assertEqual(ev.used,{0,2,4})
        self.assertEqual(ctl.released,{0,2,4})
        fresh=SpyController(.5,latency=3); fresh.step(0)
        with self.assertRaisesRegex(ValueError,'immediately before'):
            fresh.release_feedback(0,.9,2)
        with self.assertRaisesRegex(ValueError,'early'):
            fresh.release_feedback(0,.9,1)
        with self.assertRaisesRegex(ValueError,'duplicate or out-of-order'):
            ctl.release_feedback(0,.9,8)

    def test_prefix_and_unavailable_suffix_for_all_arms(self):
        pattern=np.array([.2,.9,.1,.8,.3,.7,.6,.2,.9,.1,.8,.3,.7,.6,.2,.9,.1,.8,.3,.7,.6,.2,.9,.1,.8])
        y=np.tile(pattern,2)[:50]
        z=np.arange(len(y)*3,dtype=float).reshape(-1,3)/150
        cal=np.linspace(.01,.99,250)
        for method in ('static','aci','qt','rolling','pi','history_pid','feature_pid'):
            def run(scores,cut=None,features=z):
                ctl=CausalController(.65,method=method,calibration_scores=cal if method=='aci' else None,
                                     latency=3,sparse=2,window=10,scorecaster_min=10)
                return FeedbackReplay(ctl).run(scores,z[:len(scores)],stop_after=cut)
            whole=run(y)
            truncated=run(y[:40])
            self.assertTrue(np.array_equal(whole[:40],truncated))
            corrupt=y.copy(); corrupt[40:]=np.nan
            unavailable=run(corrupt,cut=40)
            self.assertTrue(np.array_equal(whole[:40],unavailable))
        self.assertTrue(np.array_equal(paired_calibration_draw(np.arange(500),101),
                                       paired_calibration_draw(np.arange(500),101)))

    def test_scorecaster_parity_and_informative_features(self):
        rng=np.random.default_rng(42); n=50
        x=rng.uniform(0,1,n); z=np.column_stack([x,np.zeros(n),np.zeros(n)])
        scores=np.clip(.3+.5*x,0,1)
        def trajectory(method,features):
            c=CausalController(.5,method=method,eta=.02,ki=.05,latency=1,scorecaster_min=10)
            return FeedbackReplay(c).run(scores,features)
        hist=trajectory('history_pid',np.ones((n,3)))
        const=trajectory('feature_pid',np.ones((n,3)))
        self.assertTrue(np.allclose(hist,const,rtol=0,atol=1e-14))
        no_signal=trajectory('pi',z); feature=trajectory('feature_pid',z)
        self.assertTrue(np.any(np.abs(no_signal-feature)>1e-8))

    def test_pid_saturation_directions_and_hand_trace(self):
        c=CausalController(.5,method='history_pid',latency=2,sparse=2,ki=.05)
        ev=FeedbackReplay(c); ev.run([.9,.9,.1,.9,.1])
        self.assertEqual(c.thresholds[0],.5); self.assertEqual(c.thresholds[1],.5)
        self.assertAlmostEqual(c.thresholds[2],.5+.02*.9+.05*np.tan(.9*np.log(3)/6),places=14)
        self.assertEqual(c.released,{0,2})
        # Outward positive and negative saturation skips the corresponding historical integral increment.
        pos=CausalController(.99,method='pi',eta=1.,ki=.1,latency=1,anti_windup=True)
        pos_no=CausalController(.99,method='pi',eta=1.,ki=.1,latency=1,anti_windup=False)
        FeedbackReplay(pos).run([1.,1.,1.]); FeedbackReplay(pos_no).run([1.,1.,1.])
        self.assertGreaterEqual(pos.raw[1],1.)
        self.assertTrue(CausalController.outward_windup(1.1,.1))
        self.assertFalse(CausalController.outward_windup(1.1,-.1))
        neg=CausalController(.01,method='pi',eta=1.,ki=.1,latency=1,anti_windup=True)
        neg_no=CausalController(.01,method='pi',eta=1.,ki=.1,latency=1,anti_windup=False)
        FeedbackReplay(neg).run([0.,0.,0.]); FeedbackReplay(neg_no).run([0.,0.,0.])
        self.assertLessEqual(neg.raw[1],0.)
        self.assertTrue(CausalController.outward_windup(-.1,-.1))
        self.assertFalse(CausalController.outward_windup(-.1,.1))
        self.assertNotAlmostEqual(neg.raw[2],neg_no.raw[2])

    def test_scope_alignment_and_invariants(self):
        with self.assertRaisesRegex(ValueError,'scope'):
            FeedbackReplay(CausalController(.5),scope='fit').run([.5])
        with self.assertRaisesRegex(ValueError,'alignment'):
            FeedbackReplay(CausalController(.5)).run([.1,.2],[[0.]])
        c=CausalController(.5,method='pi',latency=2)
        c.step(0)
        with self.assertRaisesRegex(ValueError,'early'):
            c.release_feedback(0,.2,1)

if __name__=='__main__': unittest.main()
