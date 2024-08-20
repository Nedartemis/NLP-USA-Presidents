# run this script in the root of your project with '. ./init.sh'

# packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# python path
export PYTHONPATH="$PYTHONPATH:$(realpath .)"
export JUPYTER_PATH="$JUPYTER_PATH:$(realpath .)"
