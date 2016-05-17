FROM rcorbish/openblas-jre9

WORKDIR /home/nn

ADD run.sh  run.sh
ADD target/classes  /home/nn/classes
ADD target/dependency /home/nn/libs

RUN chmod 0500 run.sh
ENV CP classes

VOLUME [ "/home/nn/data" ]

ENTRYPOINT [ "sh", "/home/nn/run.sh" ]  
CMD [ "data/NN-TRAIN.csv", "data/NN-TEST.csv", "data/config" ]
