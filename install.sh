conda create --name cs394r python=3.7
conda activate cs394r
conda env update --name cs394r --file environment.yml

pip install git+git://github.com/bradyz/pystk.git
