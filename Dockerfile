FROM pytorch/pytorch

WORKDIR /app/

RUN apt-get update && apt-get install -y tmux && apt-get install -y numactl && apt-get install -y coreutils && apt-get install -y wget && apt-get install unzip

RUN pip install scikit-learn>=0.24.2 pyyaml>=5.4.1 numpy>=1.21.2 requests>=2.23.0

CMD ["bash"]