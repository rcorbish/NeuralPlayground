package com.rc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.BlockingQueue;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBN extends Model {
	private static Logger log = LoggerFactory.getLogger(DBN.class);
	List<String> top1000Words ;
	ResumeDataset dataset = new ResumeDataset() ;

	public DBN() {
		super() ;
	}

	public DBN( Path configDir ) throws IOException {
		super( configDir ) ;
		numInputs = 1000 ;
		top1000Words = new ArrayList<>() ;
	}

	protected List<String> getTop1000Words() {
		return top1000Words ;		
	}

	@SuppressWarnings("unchecked")
	public BlockingQueue<String> train( Path trainingData ) throws Exception {
		StreamIterationListener sil = new StreamIterationListener(100) ;

		forEach( mln -> mln.setListeners(sil) ) ;

		log.info("Load data from " + trainingData );

		RecordReader recordReader = new CSVRecordReader(1);
		// Point to data path. 
		recordReader.initialize(new FileSplit(trainingData.toFile()));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 200, 0, numOutputs);

		Runnable r = new Runnable() {
			public void run() {
				log.info("Train model....");

				while(iter.hasNext()) {
					DataSet ds = iter.next();
					ds.normalizeZeroMeanZeroUnitVariance();
					getModel( 0 ).fit( ds ) ;
				}		
				log.info("Training done.");
			}
		} ;
		new Thread( r ).start(); 
		return sil.getStream() ;
	}


	public String test( Path testData ) throws Exception {

		RecordReader recordReader = new CSVRecordReader(1);

		log.info("Load verification data from " + testData ) ;
		// Point to data path. 
		recordReader.initialize(new FileSplit(testData.toFile()));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 200, 0, numOutputs );

		Evaluation eval = new Evaluation( numOutputs );

		while(iter.hasNext()) {
			DataSet ds = iter.next();
			ds.normalizeZeroMeanZeroUnitVariance();
			INDArray predict2 = getModel(0).output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
			eval.eval(ds.getLabels(), predict2);
		}
		log.info(eval.stats());
		log.info("All Done");

		return eval.stats() ;
	}


	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		this.numInputs = numInputs ;
		this.numOutputs = numOutputs ;

		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(400)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0)
				.momentum(0.5)
				.momentumAfter(Collections.singletonMap(3, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.list(numLayers) ;
		for( int i=0 ; i<(numLayers-1) ; i++ ) {
			lb.layer(i, 
				new RBM.Builder().nIn(numInputs).nOut(numInputs)
				.weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
				.visibleUnit(RBM.VisibleUnit.BINARY)
				.hiddenUnit(RBM.HiddenUnit.BINARY)
				.build() )
				;
		}
		MultiLayerConfiguration conf = lb
				.layer(numLayers-1, 
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                .nIn(numInputs).nOut(numOutputs).build() ) 
				.pretrain(true).backprop(false)
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


