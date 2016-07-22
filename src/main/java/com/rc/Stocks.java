package com.rc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Stocks {

	private String[] tickers = new String[] { "BAC", "GE", "RAD", "WFC", "C", "CHK", "PFE", "AA", "KMI", 
			"DAL", "RIG", "KO", "XOM", "MET", "MS", "GS", "V", "WMT",   
			"MSFT", "AAPL", "INTC", "QQQ", "NVDA", "CSX", "FEYE", "WDC", "AMZN", "AGNC"
	} ;
	private static Logger log = LoggerFactory.getLogger(Stocks.class);

	private LSTM nn ;

	public Stocks() {
		nn = new LSTM() ; 
		nn.createModelConfig( 4, tickers.length , 1 ) ;
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
					Thread.sleep(10000);
				} catch (InterruptedException e) {
				}
				log.info( "Reading prices for {}", ticker ) ;
				List<float[]> px = dip.downloadedData( ticker );
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
			Map<String, List<float[]> > prices = load() ;
			
			float min = Float.MAX_VALUE ;
			float max = Float.MIN_VALUE ;
			
			for( String ticker : prices.keySet() ) {
				for( float days[] : prices.get( ticker ) ) {
					for( float fl : days ) {
						min = Math.min( min, fl ) ;
						max = Math.max( max, fl ) ;
					}					
				}
			}
			
			float scale = 1.f / ( max - min ) ;
			
			for( int start=0 ; start<189 ; start++ ) {
				for( int seq = 0 ; seq<10 ; seq++ ) {
					sb.setLength(0);
					for( String ticker : prices.keySet() ) {
						List<float[]> data = prices.get( ticker ) ;
						float d[] = data.get(seq+start) ;
						sb.append( ( d[1] -min ) * scale ).append(',') ;
//						sb.append( d[2] ).append(',') ;
//						sb.append( d[3] ).append(',') ;
//						sb.append( d[4] ).append(',') ;
					}
					
					List<float[]>data = prices.get( outputTicker ) ;
					float d[] = data.get(1+seq+start) ;
					sb.append( ( d[1] -min ) * scale ).append( ',') ;
					fw.append( sb ) ;
				}
				fw.append( "\n" ) ;
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

	public void predict() throws Exception {
		String rc = nn.test( null ) ;
		log.info( rc ) ; 
	}

	public static void main( String args[] ) {
		Stocks self = new Stocks() ;
		try {
			Nd4j.ENFORCE_NUMERICAL_STABILITY = true ;
			self.train( "MSFT" ) ;
			self.predict();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
