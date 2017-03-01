package com.rc;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit;
import org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBNA extends Model {
	private static Logger log = LoggerFactory.getLogger(DBNA.class);

	public DBNA() {
		super() ;
	}

	public DBNA( Path configDir ) throws IOException {
		super( configDir ) ;
	}

	public BlockingQueue<String> train( Path trainingData ) throws Exception {
		StreamIterationListener sil = new StreamIterationListener(100) ;
		forEach( mln -> mln.setListeners(sil) ) ;

		log.info("Load data from " + trainingData );

		org.datavec.api.records.reader.RecordReader recordReader = new org.datavec.api.records.reader.impl.csv.CSVRecordReader(1);
		// Point to data path. 
		recordReader.initialize(new org.datavec.api.split.FileSplit(trainingData.toFile()));
		org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(recordReader, 250, 0, numInputs);

		Runnable r = new Runnable() {
			@Override
			public void run() {
				try {
					log.info("Train model....");

					while(iter.hasNext()) {
						DataSet ds = iter.next();
						ds.normalizeZeroMeanZeroUnitVariance();
						getModel( 0 ).fit( new DataSet( ds.getFeatureMatrix(), ds.getFeatureMatrix() ) ) ;
					}		
					log.info("Training done.");
					sil.getStream().offer( "Training done.");
				} catch( IllegalStateException er ) {
					log.error( "Error during training", er ) ;;
					sil.getStream().offer( er.getMessage() + "<br><br>Check number of inputs & outputs for compatability with your data." ) ;
					er.printStackTrace();
				} finally {
					log.info( "Finally - we close the result stream." ) ;
					sil.getStream().offer( "" ) ;
				}
			}
		} ;
		new Thread( r ).start(); 
		return sil.getStream() ;
	}


	public String test( Path testData ) throws Exception {

		Evaluation eval = null ;
		if( testData != null ) {
			org.datavec.api.records.reader.RecordReader recordReader = new org.datavec.api.records.reader.impl.csv.CSVRecordReader(1);
			// Point to data path. 
			recordReader.initialize(new org.datavec.api.split.FileSplit( testData.toFile() ) );
			org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(recordReader, 250, 0, numInputs);

			eval = new Evaluation( numInputs );

			while(iter.hasNext()) {
				DataSet ds = iter.next();
				ds.normalizeZeroMeanZeroUnitVariance();
				INDArray predict2 = getModel(0).output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
				eval.eval(ds.getFeatureMatrix(), predict2);
			}
			log.info(eval.stats());
			log.info("All Done");

		}
		return eval.stats() ;
	}

	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		this.numInputs = numInputs ;
		this.numOutputs = numInputs ;
	
		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(1000)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.list() ;
		for( int i=0 ; i<(numLayers-1) ; i++ ) {
			lb.layer( i, 
					new RBM.Builder().
					nIn(numInputs).
					nOut(numInputs).
					lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).
					activation("relu").
					weightInit(WeightInit.RELU).
					hiddenUnit(HiddenUnit.GAUSSIAN).
					visibleUnit(VisibleUnit.GAUSSIAN).						
					build()
					) ;
		}
		MultiLayerConfiguration conf = lb
				.layer(numLayers-1, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
						.nIn(numInputs).nOut(numInputs).build())
				.pretrain(true).backprop(true)
				.build();
		addModel( conf ) ;
	}

}


