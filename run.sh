

export CP="classes"
for l in libs/* 
do 
	CP=$CP:$l 
done 

java -cp $CLASSPATH:$CP com.rc.Run $@
