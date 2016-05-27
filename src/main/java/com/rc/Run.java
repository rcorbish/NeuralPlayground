package com.rc;

import java.io.File;
import java.io.IOException;

public class Run {

	public static void main(String[] args) throws IOException {
		if( args.length < 1 ) {
			System.err.println( "Missing arguments.");
			System.exit( -1 );
		}
		
		File trainDataFile = new File( args[0] ) ;
		File testDataFile = args.length>1 ? new File( args[1] ) : null ;
		File configDir = args.length>2 ? new File( args[2] ) : null ;
		Run self = new Run( trainDataFile, testDataFile, configDir ) ;
		new WebServer( self.nn ) ;
		self.run( args[0], args.length>1 ? args[1] : null ) ;
	}
	
	private Model nn ;
	private File configDir ;
	public Run( File trainingData, File testData, File configDir ) throws IOException {
//		nn = new MultiLayer( trainingData, testData, configDir ) ;
		nn = new DBN( trainingData, testData, configDir ) ;
	}
	
	public void run( String trainingData, String testData ) {
		try {						
			nn.loadModel( true ) ;
			
			if( nn.getModel() == null ) {
				nn.setModel( nn.createModelConfig() ) ;
			}
			
			nn.train();

			nn.saveModel( true);

			if( testData != null ) {
				nn.test();
			}
		} catch( Throwable t ) {
			t.printStackTrace();
		}
	}

}
