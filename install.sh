cur_DIR=$(pwd)
DIR=$(dirname $0)
cd $DIR
python setup.py sdist
pip install dist/RoomSimulator-1.0.tar.gz
cd $cur_DIR
