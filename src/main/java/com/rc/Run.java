package com.rc;

import java.io.File;

import org.nd4j.linalg.dataset.DataSet;

public class Run {

	public static void main(String[] args) {
		try {			
			
			File trainDataFile = new File( args[0] ) ;
			//DBN nn = new DBN() ;
			MultiLayer nn = new MultiLayer() ;
			
			if( args.length > 2 ) {
				File saveDir = new File( args[2] ) ;
				try {
					nn.loadModel(saveDir, true ) ;
				} catch( Throwable t ) {
					System.err.print( t.getLocalizedMessage() ) ;
				}
			} 
			
			if( nn.getModel() == null ) {
				nn.setModel( nn.createModelConfig(trainDataFile) ) ;
			}
			
			DataSet ds = nn.train( trainDataFile );
//			if( ds != null ) {
//				nn.plot(ds);
//			}

			if( args.length > 2 ) {
				File saveDir = new File( args[2] ) ;
				nn.saveModel(saveDir, true);
			}

			if( args.length > 1 ) {
				File testDataFile = new File( args[1] ) ;
				nn.test(testDataFile);
			}
		} catch( Throwable t ) {
			t.printStackTrace();
		}
	}

}
