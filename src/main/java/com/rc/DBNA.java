package com.rc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.collections.bag.HashBag;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBNA extends Model {
	private static Logger log = LoggerFactory.getLogger(DBNA.class);
	List<String> top1000Words ;
	ResumeDataset dataset = new ResumeDataset() ;
	
	public DBNA() {
		super() ;
	}
	
	public DBNA(Path trainingData, Path testData, Path configDir ) throws IOException {
		super( trainingData, testData, configDir ) ;
		numInputs = 1000 ;
		top1000Words = new ArrayList<>() ;
	}

	protected List<String> getTop1000Words() {
		return top1000Words ;		
	}

	@SuppressWarnings("unchecked")
	public void train() throws Exception {

		if( this.trainingData != null ) {
			dataset.getDatasetIterator( this.trainingData ) ;
		
			getModel(0).setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(10)));

			log.info("Load data from " + trainingData );

			DataSetIterator iter = dataset.getDatasetIterator( this.trainingData ) ;

			log.info("Train model....");
			while(iter.hasNext()) {
				DataSet next = iter.next();
				getModel(0).fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
			}
		}
	}


	public Evaluation test() throws Exception {

		Evaluation eval = null ;
		if( this.testData != null ) {
			DataSetIterator iter = dataset.getDatasetIterator( this.testData ) ;

			eval = new Evaluation( numInputs );
	
			int n = 0 ;
			try {
				while(iter.hasNext()) {
					n++ ;
					DataSet ds = iter.next();
					INDArray predict2 = getModel(0).output( ds.getFeatureMatrix(), false );
					for( int r=0 ; r<predict2.rows() ; r++ ) {
						INDArray row = predict2.getRow(r) ;
						StringBuilder sb = new StringBuilder() ;
						for( int c=0 ; c<row.columns() ; c++ ) {
							int j = (int)(1.0e4 * row.getDouble(c) );
							if( j>1 ) {
								sb.append( getTop1000Words().get(c) ).append( '\n' ) ;
							}
						}
					//eval.eval( ds.getFeatureMatrix(), predict2 ) ;
						log.info( sb.toString() ) ;
					}
				}
			} catch( Throwable t ) {
				t.printStackTrace();
				log.info( "Managed to load " + n + " items" );
			}
		}
		return eval ;
	}

	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(100)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.list(numLayers) ;
		for( int i=0 ; i<(numLayers-1) ; i++ ) {
			lb.layer(i, new RBM.Builder().nIn(numInputs).nOut(numInputs).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) ;
		}
		MultiLayerConfiguration conf = lb
				.layer(numLayers-1, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
											.nIn(numInputs).nOut(numInputs).build())
				.pretrain(true).backprop(true)
				.build();
		addModel( conf ) ;
	}



	public void loadModel( boolean loadUpdater ) throws IOException, ClassNotFoundException {
		super.loadModel(loadUpdater);
		
		if( configDir != null ) {
			Path p = configDir.resolve( "top1000.txt" ) ;
			try( BufferedReader br = Files.newBufferedReader(p ) ) {
				List<String> l = getTop1000Words() ;
				l.clear(); 
				for( String s=br.readLine() ; s!=null ; s=br.readLine() ) {
					l.add(s) ;
				}
				numInputs = l.size() ;
			}
		}
	}

	public void saveModel( boolean saveUpdater ) throws IOException {
		super.saveModel( saveUpdater ) ;

		if( configDir != null ) {
			Path p = configDir.resolve( "top1000.txt" ) ;
			try( Writer w = Files.newBufferedWriter( p ) ) {
				for( String s : getTop1000Words() ) {
					w.append( s ).append( System.getProperty("line.separator") ) ;
				}
				w.flush();
			}

		}
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


