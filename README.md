# Code samples for "Neural Networks and Deep Learning"

This repository contains code samples for my book on ["Neural Networks
and Deep Learning"](http://neuralnetworksanddeeplearning.com).

This checkout has been patched to run on Apple Silicon Macs.

## Quick start (macOS / Apple Silicon)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`python3.12` supports the NumPy / SciPy / scikit-learn scripts.
The Theano-based code (`src/network3.py`, `src/conv.py`) requires
`python3.10` or `python3.11` because `Theano-PyMC` does not build on 3.12.

For Theano-based runs:

```bash
python3.10 -m venv .venv-theano
source .venv-theano/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run simple baselines:

```bash
cd src
python mnist_average_darkness.py
python mnist_svm.py
```

`src/network3.py` and `src/conv.py` target `Theano-PyMC`.

As the code is written to accompany the book, I don't intend to add
new features. However, bug reports are welcome, and you should feel
free to fork and modify the code.

## License

MIT License

Copyright (c) 2012-2022 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
