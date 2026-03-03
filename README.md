* Neural Networks and Deep Learning
This repository contains updated and modified code samples based on the book **"Neural Networks and Deep Learning"** by Michael Nielsen.

## Overview

This version is a modernized fork of the original repository. The codebase has been:

- Fully updated for **Python 3.10**
- Modified to remove deprecated dependencies

---

## Key Changes from the Original Repository

- Migrated from **Python 2.6/2.7** to **Python 3.10**
- Removed legacy Python 2 syntax
- Updated deprecated library usage

---

## About `network3.py`

The original implementation depended on older versions of **Theano (0.6/0.7)**, which are no longer maintained.

In this repository:

- Theano dependencies have been replaced.
- The implementation has been adapted to use modern numerical computation libraries.

---

## Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd <repo-name>
```

Create a virtual environment (recommended):

```bash
python3.10 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```