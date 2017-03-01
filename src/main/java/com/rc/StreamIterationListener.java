package com.rc;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("serial")
public class StreamIterationListener  implements IterationListener {
    private int printIterations = 10;
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationListener.class);
    private boolean invoked = false;

    BlockingQueue<String> buffer ;
    
    public StreamIterationListener(int printIterations) {
    	this() ;
        this.printIterations = printIterations;
        
    }

    public StreamIterationListener() {
    	buffer = new ArrayBlockingQueue<>(200) ;
    }

    public BlockingQueue<String> getStream() { return buffer ; }
    
    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(printIterations <= 0)
            printIterations = 1;
        if( (iteration % printIterations) == (printIterations-1) ) {
            invoke();
            double result = model.score();
            try {
            	buffer.add( "Score at iteration " + iteration + " is " + result ) ;
            } catch( Throwable ex ) {
            	// ignored - if queue is full we'll miss a message
            }
        }
    }
}