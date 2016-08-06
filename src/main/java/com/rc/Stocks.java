package com.rc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Array;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.util.Collections;

public class Stocks {

	private String[] tickers = new String[] {
//			"BAC", "GE", "RAD", "WFC", "C", "CHK", "PFE", "AA", "KMI", 
//			"DAL", "RIG", "KO", "XOM", "MET", "MS", "GS", "V", "WMT",   
//			"MSFT", "AAPL", "INTC", "NVDA", "CSX", 
//			"FEYE", 
//			"WDC", 
//			"AMZN", "AGNC", "QQQ" 
			"SVXY", "UVXY"
	} ;
	//private String[] tickers = new String[] { "SVXY", "UVXY" } ;
	
	
	private static Logger log = LoggerFactory.getLogger(Stocks.class);

	public static int TIME_SERIES_LENGTH = 12 ;
	public static int HISTORY_LENGTH = 200 ;
	
	private LSTM nn ;
	private Map<String, List<float[]> > prices ;

	public Stocks() throws ClassNotFoundException, IOException {
		nn = new LSTM() ; 
		nn.createModelConfig( 4, tickers.length, 1 ) ;		
		prices = load() ;
	}



	@SuppressWarnings("unchecked")
	public Map<String, List<float[]> > load() throws IOException, ClassNotFoundException {
		Map<String, List<float[]> > rc = null ;

		File tmp = new File( "/home/richard/px.dat" ) ;
		Calendar cal = Calendar.getInstance() ;
		cal.add(Calendar.DATE, -1 ) ;
		if( cal.get( Calendar.DAY_OF_WEEK ) == Calendar.SUNDAY ) cal.add(Calendar.DATE, -1 );
		if( cal.get( Calendar.DAY_OF_WEEK ) == Calendar.SATURDAY ) cal.add(Calendar.DATE, -1 );

		if( tmp.lastModified() < cal.getTimeInMillis() ) {
			DownloadIndexPrices dip = new DownloadIndexPrices() ;
			rc = new HashMap<>() ;

			for( String ticker : tickers ) {
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
				}
				log.info( "Reading prices for {}", ticker ) ;
				List<float[]> px = dip.downloadedData( HISTORY_LENGTH, ticker );
				rc.put( ticker,  px ) ;
			}
			try( FileOutputStream os = new FileOutputStream(tmp) ; ObjectOutputStream oos = new ObjectOutputStream(os); ) {
				oos.writeObject( rc ); 
			}
		}
		if( rc == null ) {
			try( FileInputStream is = new FileInputStream(tmp) ; ObjectInputStream ois = new ObjectInputStream(is); ) {
				rc = (Map<String, List<float[]>>)ois.readObject() ; 
			}
		}
		return rc ;
	}


	public void train( String outputTicker ) throws Exception {
		File f = File.createTempFile( "stox", ".dat" ) ;
		StringBuilder sb = new StringBuilder() ;
		try( FileWriter fw = new FileWriter( f ) ) {
			
			int numTestDatasets = (HISTORY_LENGTH-TIME_SERIES_LENGTH-1) ;
			
			List<Integer> solution = new ArrayList<>();
			for (int i = 0; i < numTestDatasets ; i++) {
			    solution.add(i);
			}
			java.util.Collections.shuffle(solution);
			
			for( int i=0 ; i<numTestDatasets ; i++ ) {
				int start = solution.get(i) ;
				start = i ;
				for( int seq = 0 ; seq<TIME_SERIES_LENGTH ; seq++ ) {
					sb.setLength(0);
					for( String ticker : prices.keySet() ) {
						List<float[]> data = prices.get( ticker ) ;
						float d[] = data.get(seq+start) ;
						sb.append( d[1] ).append(',') ;
					}				
					List<float[]>data = prices.get( outputTicker ) ;
					float d[] = data.get(1+seq+start) ;
					sb.append( d[1] ).append( ',') ;
					fw.append( sb ) ;
				}
				fw.append( '\n' ) ;				
			}
		} catch (Exception e) {
			log.error( "Failed to load data", e );
		}

		Path p = f.toPath() ;
		BlockingQueue<String> status = nn.train( p ) ;

		String info = status.take() ;
		while( info.length()>0 ) {
			log.info( info ); 
			info = status.take() ;
		}
	}

	public void predict( String outputTicker ) throws Exception {

		nn.getModel(0).rnnClearPreviousState();

		INDArray input  = Nd4j.create(1, tickers.length, 1 ) ; 

		int ix[] = new int[] { 0, 0, 0 } ;

		for( int i=0 ; i<TIME_SERIES_LENGTH ; i++ ) {
			ix[1] = 0 ;
			for( String ticker: tickers ) {			
				List<float[]> data = prices.get( ticker ) ;			
				float d[] = data.get( data.size() - TIME_SERIES_LENGTH + i ) ;			
				input.putScalar( ix, d[1] ) ;
				ix[1] ++ ;
			}
			nn.getModel(0).rnnTimeStep(input) ;
		}
		INDArray output = nn.getModel(0).output( input ) ;

		log.info( "Tomorrow's px for {} is {} " , outputTicker, output.getDouble( 0 ) ) ;
	}

	public static void main( String args[] ) {
		try {
			Stocks self = new Stocks() ;
			Nd4j.ENFORCE_NUMERICAL_STABILITY = true ;
			self.train( "UVXY" ) ;
			self.predict( "UVXY" );
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
