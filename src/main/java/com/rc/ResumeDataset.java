package com.rc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ResumeDataset {
	private static Logger log = LoggerFactory.getLogger(ResumeDataset.class);
	List<String> top1000Words ;

	protected List<String> getTop1000Words() {
		return top1000Words ; 
	}

	public org.nd4j.linalg.dataset.api.iterator.DataSetIterator getDatasetIterator(Path data ) throws IOException {
		top1000Words = new ArrayList<>() ;

		Map<String,Integer> dict = new HashMap<>() ;

		List<String> lines = Files.readAllLines( data ) ;

		for( String s : lines ) {
			Collection<String> processedWords = preprocessText( s ) ;
			for( String w : processedWords ) {
				Integer nw = dict.get( w ) ;
				if( nw == null ) {
					dict.put( w, 1 ) ;
				} else {
					dict.put( w, 1+nw ) ;
				}
			}
		}

		List<String> popularTexts = dict.keySet().stream() 
		.filter( p -> dict.get(p)>1 )
		.sorted( new Comparator<String>() {
			public int compare(String o1, String o2) {
				return dict.get(o1) - dict.get(o2) ;
			}
		})
		.limit(1000)
		.collect( Collectors.toList() ) 
		;
		
		top1000Words = popularTexts ;

		log.info( "Training with {} inputs", top1000Words.size() ) ;
		
		List<DataSet> al = new ArrayList<>( lines.size() ) ;

		for( String s : lines ) {
			INDArray features = Nd4j.create(top1000Words.size()) ;
			Collection<String> preprocessedWords = preprocessText( s ) ;
			for( String t : preprocessedWords ) {
				int ix = top1000Words.indexOf( t ) ;
				if( ix>0 ) {
					features.putScalar(ix, 1 ) ;
				}
			}
			DataSet ds = new DataSet( features, features ) ;
			al.add(ds) ;
		}
		org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new ListDataSetIterator( al, 100 ) ;
		return iter ;
	}

	protected Collection<String> preprocessText( String line ) {
		List<String> rc = new ArrayList<>() ;
		
		for( String t : line.trim().split("\\s" ) ) {
			if( t.trim().length() > 0 ) {
				if( t.matches( "[\\d]+" ) ) {
					t ="**NUMBER**" ;
				} else if( t.charAt(0) == '@' ) {
					t ="**TWITTER**" ;
				} else if( t.indexOf('@')>0 ) {
					t ="**EMAIL**" ;
				} else if( t.indexOf("http")==0 || t.indexOf('/')>0 ) {
					t ="**URL**" ;
				} else if( t.matches( "[\\s]+" ) ) {
					continue ;
				}
				rc.add( t ) ;
			}
		}
		return rc ;
	}
}


