## Project Overview

The goal of this project is to generate activations from a Torch model. These
activations will then be usable in [ZML](https://github.com/zml/zml).

### Prerequisites

- Python 3.11

## Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/vctrmn/zml-torch-activation-example.git
```

2. Create a python virtual env

```
python3 -m venv venv
```

3. Activate the virtual env

```
. venv/bin/activate
```

4. Install the dependencies

```
pip install -r requirements.txt
```

## Usage

1. Activate the virtual env

```
. venv/bin/activate
```

2. Run the activation script you want

```
python -m src.modern_bert
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
