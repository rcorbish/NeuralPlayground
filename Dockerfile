FROM rcorbish/openblas-jre9

WORKDIR /home/nn

ADD run.sh  run.sh
ADD src/main/resources  /home/nn/resources
ADD target/classes  /home/nn/classes
ADD target/dependency /home/nn/libs

RUN chmod 0500 run.sh ; \
	sed -i "s/\${VERSION}/$(date)/g" resources/templates/layout.jade
ENV CP classes:resources

VOLUME [ "/home/nn/data" ]

ENTRYPOINT [ "sh", "/home/nn/run.sh" ]  
CMD [ "data/config" ]
