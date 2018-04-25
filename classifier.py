import numpy as np
import pandas as pd

def classify(clf, averages):
    def format_averages(averages):
        pairs = []
        times = []
        for key in TOP_PAIRS:
            delta = 0
            if key in averages:
                delta = averages[key][0]
            pairs.append(key)
            times.append(delta)
        d = {
            "pair" : pairs,
            "delta_avg" : times
        }
        df = pd.DataFrame(data = d)
        df = df[["pair", "delta_avg"]]
        df = df.sort_values("pair", ascending=False)
        df = df.transpose()
        a = df.iloc[1:]
        a = pd.DataFrame(a)
        a = a.fillna(0)
        a = np.asmatrix(a)
        return a
    
    averages = format_averages(averages)
    if averages.shape[1] > 10:
        return clf.predict(averages)
    else:
        return None

TOP_PAIRS = {"'.',Key.space",
 "'a','c'",
 "'a','l'",
 "'a','n'",
 "'a','r'",
 "'a','s'",
 "'a','t'",
 "'a',Key.space",
 "'b','e'",
 "'c','a'",
 "'c','e'",
 "'c','h'",
 "'c','o'",
 "'d','d'",
 "'d','e'",
 "'d',Key.space",
 "'e','a'",
 "'e','c'",
 "'e','d'",
 "'e','e'",
 "'e','l'",
 "'e','n'",
 "'e','r'",
 "'e','s'",
 "'e','t'",
 "'e',Key.space",
 "'f',Key.space",
 "'g',Key.space",
 "'h','a'",
 "'h','e'",
 "'h','i'",
 "'i','n'",
 "'i','o'",
 "'i','s'",
 "'i','t'",
 "'k','e'",
 "'l','a'",
 "'l','e'",
 "'l','i'",
 "'l','l'",
 "'l',Key.space",
 "'m','a'",
 "'m','e'",
 "'n','d'",
 "'n','e'",
 "'n','g'",
 "'n','t'",
 "'n',Key.space",
 "'o','f'",
 "'o','m'",
 "'o','n'",
 "'o','r'",
 "'o','u'",
 "'o',Key.space",
 "'r','a'",
 "'r','e'",
 "'r','i'",
 "'r','o'",
 "'r',Key.space",
 "'s','e'",
 "'s','t'",
 "'s',Key.space",
 "'t','a'",
 "'t','e'",
 "'t','h'",
 "'t','i'",
 "'t','o'",
 "'t',Key.space",
 "'u','r'",
 "'u','t'",
 "'v','e'",
 "'y',Key.space",
 'Key.backspace,Key.backspace',
 'Key.backspace,Key.shift_r',
 "Key.cmd,'ç'",
 "Key.cmd,'√'",
 'Key.cmd,Key.tab',
 'Key.enter,Key.enter',
 'Key.left,Key.left',
 'Key.right,Key.right',
 'Key.shift_r,Key.enter',
 "Key.space,'a'",
 "Key.space,'b'",
 "Key.space,'c'",
 "Key.space,'d'",
 "Key.space,'f'",
 "Key.space,'h'",
 "Key.space,'i'",
 "Key.space,'l'",
 "Key.space,'m'",
 "Key.space,'o'",
 "Key.space,'p'",
 "Key.space,'r'",
 "Key.space,'s'",
 "Key.space,'t'",
 "Key.space,'w'",
 "Key.space,'y'",
 'Key.space,Key.backspace',
 'Key.space,Key.shift',
 'Key.space,Key.shift_r'}
