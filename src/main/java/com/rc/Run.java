package com.rc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Run {
	private static Logger log = LoggerFactory.getLogger(Run.class);

	public static void main(String[] args) throws IOException {
		if( args.length < 1 ) {
			System.err.println( "Missing arguments.");
			System.exit( -1 );
		}
		
		Path trainDataFile = Paths.get( args[0] ) ;
		Path testDataFile = args.length>1 ? Paths.get( args[1] ) : null ;
		Path configDir = args.length>2 ? Paths.get( args[2] ) : null ;
		Run self = new Run( trainDataFile, testDataFile, configDir ) ;
		new WebServer( ) ;
		//self.run( args[0], args.length>1 ? args[1] : null ) ;
	}
	
	private Model nn ;
	private File configDir ;
	public Run( Path trainingData, Path testData, Path configDir ) throws IOException {
//		nn = new MultiLayer( trainingData, testData, configDir ) ;
		nn = new DBN( trainingData, testData, configDir ) ;
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
			nn.train();
			
			log.info( "Saving model" ) ;			
			nn.saveModel( true);

			log.info( "Testing model" ) ;			
			nn.test();
		} catch( Throwable t ) {
			t.printStackTrace();
		}
	}

}
