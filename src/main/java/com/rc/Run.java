package com.rc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Run {
	private static Logger log = LoggerFactory.getLogger(Run.class);

	public static void main(String[] args) throws IOException {

		if( args.length < 1 ) {
			System.err.println( "Missing arguments.");
			System.exit( -1 );
		}
		
		Path configDir = args.length>2 ? Paths.get( args[0] ) : null ;
		Run self = new Run( configDir ) ;
		new WebServer( ) ;
	}
	
	private Model nn ;
	private File configDir ;
	public Run( Path configDir ) throws IOException {
	}
	
	public void run( String trainingData, String testData ) {
		try {			
			log.info( "Loading model" ) ;
			try { 
				nn.loadModel( true ) ;
			} catch( Throwable t ) {
				log.info( "Can't load model." );
			}
			
			log.info( "Training model" ) ;			
			nn.train( Paths.get(trainingData) );
			
			log.info( "Saving model" ) ;			
			nn.saveModel( true);

			log.info( "Testing model" ) ;			
			nn.test( Paths.get(testData) );
		} catch( Throwable t ) {
			t.printStackTrace();
		}
	}

}


