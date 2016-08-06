package com.rc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.net.ssl.HttpsURLConnection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DownloadIndexPrices {
	private static Logger log = LoggerFactory.getLogger(DownloadIndexPrices.class);

	private static String IndexUrlName = "https://fred.stlouisfed.org/categories/32255/downloaddata/STOCKMARKET_csv_2.zip" ;
	private static String QuoteUrlName = "http://ichart.finance.yahoo.com/table.csv?s=" ;

	public List<float[]> downloadedData( int historyLength,  String ticker ) throws IOException {

		int numLines = historyLength ;
		URL indexUrl = new URL( QuoteUrlName + ticker ) ;

		List<float[]> rc = new ArrayList<>() ;
		
		HttpURLConnection indexConnection = (HttpURLConnection)indexUrl.openConnection() ;	
		indexConnection.setRequestProperty("User-Agent", "LSTM");
		try( InputStream is = indexConnection.getInputStream() ; Reader r = new InputStreamReader(is) ; BufferedReader br = new BufferedReader(r) ; ) {		
			String s = br.readLine() ;
			SimpleDateFormat sdf = new SimpleDateFormat( "yyyy-MM-dd" ) ;
			while( (s=br.readLine()) != null ) { // read data lines
				if( --numLines == 0 ) break ;
				String cols[] = s.split( "," ) ;
				float dt = Float.parseFloat( cols[0].replaceAll( "-", "" ) ) ; 
				float high = Float.parseFloat( cols[2] ) ; 
				float low = Float.parseFloat( cols[3] ) ;
				float close = Float.parseFloat( cols[4] ) ;
				float volume = Float.parseFloat( cols[5] ) ;
				rc.add( new float[] { dt, close, high, low, volume } ) ;
			}
		} finally {
			indexConnection.disconnect();
		}
		return rc ;
	}


	public void downloadedFedData() throws IOException {

		URL indexUrl = new URL( IndexUrlName ) ;

		HttpsURLConnection indexConnection = (HttpsURLConnection)indexUrl.openConnection() ;		

		byte fileBuffer[] = new byte[8192] ;
		try( 
				InputStream is = indexConnection.getInputStream() ;
				ZipInputStream zis = new ZipInputStream( is ) ) {  // end auto-close section

			ZipEntry entry ;
			while( (entry = zis.getNextEntry()) != null ) {		// read each file entry
				String name = entry.getName() ;
				if( name.endsWith(".csv") ) {					// only look at CSV files
					long size = entry.getCompressedSize() ;

					log.info( "Reading {} bytes from {}.", size, name ) ;

					for( int n = zis.read( fileBuffer ) ; n>0 ; n = zis.read( fileBuffer ) ) { // read data byte by byte 
						String s = new String( fileBuffer, Charset.forName("UTF8") ) ;
						log.info( s ) ;
					}
				}
			}
		} finally {
			indexConnection.disconnect();
		}
	}
}
