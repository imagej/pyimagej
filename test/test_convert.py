import unittest
from imagej.convert import to_java, to_python

class TestConvert(unittest.TestCase):

    def testList(self):
        l = 'The quick brown fox jumps over the lazy dogs'.split()
        jl = to_java(l)
        for e, a in zip(l, jl):
            self.assertEqual(e, to_python(a))
        pl = to_python(jl)
        self.assertEqual(l, pl)

    def testSet(self):
        s = set(['orange', 'apple', 'pineapple', 'plum'])
        js = to_java(s)
        self.assertEqual(len(s), js.size())
        for e in s:
            self.assertTrue(js.contains(to_java(e)))
        ps = to_python(js)
        self.assertEqual(s, ps)

    def testDict(self):
        d = {
            'access_log': [
                {'stored_proc': 'getsomething'},
                {'uses': [
                    {'usedin': 'some->bread->crumb'},
                    {'usedin': 'something else here'},
                    {'stored_proc': 'anothersp'}
                ]},
                {'uses': [
                    {'usedin': 'blahblah'}
                ]}
            ],
            'reporting': [
                {'stored_proc': 'reportingsp'},
                {'uses': [{'usedin': 'breadcrumb'}]}
            ]
        }
        jd = to_java(d)
        self.assertEqual(len(d), jd.size())
        for k, v in d.items():
            jk = to_java(k)
            self.assertTrue(jd.containsKey(jk))
            self.assertEqual(v, to_python(jd.get(jk)))
        pd = to_python(jd)
        self.assertEqual(d, pd)

    def testMixed(self):
        d = {'a':'b', 'c':'d'}
        l = ['e', 'f', 'g', 'h']
        s = set(['i', 'j', 'k'])

        # mixed types in a dictionary
        md = {'d': d, 'l': l, 's': s, 'str': 'hello'}
        jmd = to_java(md)
        self.assertEqual(len(md), jmd.size())
        for k, v in md.items():
            jk = to_java(k)
            self.assertTrue(jmd.containsKey(jk))
            self.assertEqual(v, to_python(jmd.get(jk)))
        pmd = to_python(jmd)
        self.assertEqual(md, pmd)

        # mixed types in a list
        ml = [d, l, s, 'hello']
        jml = to_java(ml)
        for e, a in zip(ml, jml):
            self.assertEqual(e, to_python(a))
        pml = to_python(jml)
        self.assertEqual(ml, pml)

if __name__ == '__main__':
    unittest.main()
