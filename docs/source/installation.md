
# Installation

## PyPI (recommended)

The Python package is hosted on the [Python Package Index (PyPI)](https://pypi.org/project/nbrefactor/).

The latest published version of **nbrefactor** can be installed using

```bash
pip install --update nbrefactor
```

## Manual Installation

Simply clone the repo and extract the files in the `nbrefactor` folder,
then run:

```bash
pip install -r requirements.txt
pip install -e .
```

Or use one of the scripts below:

### GIT

- `cd` into your project directory
- Use `sparse-checkout` to pull the library files only into your project directory
  ```bash
  git init nbrefactor
  cd nbrefactor
  git remote add -f origin https://github.com/ThunderStruct/nbrefactor.git
  git config core.sparseCheckout true
  echo "nbrefactor/*" >> .git/info/sparse-checkout
  git pull --depth=1 origin master
  pip install -r requirements.txt
  pip install -e .
  ```

### SVN

- `cd` into your project directory
- `checkout` the library files
  ```bash
  svn checkout https://github.com/ThunderStruct/nbrefactor/trunk/nbrefactor
  pip install -r requirements.txt
  pip install -e .
  ```
