package com.rc;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultiLayer extends Model {

	private static Logger log = LoggerFactory.getLogger(MultiLayer.class);

	public MultiLayer(Path configDir ) throws IOException {
		super( configDir ) ;
	}
	
	public MultiLayer() {
		super() ;
	}
	
	@Override
	public BlockingQueue<String> train( Path trainingData ) throws Exception {

		StreamIterationListener sil = new StreamIterationListener(100) ;
		forEach( mln -> mln.setListeners(sil) ) ;

		log.info("Load data from " + trainingData );

		RecordReader recordReader = new CSVRecordReader(1);
		// Point to data path. 
		recordReader.initialize(new FileSplit(trainingData.toFile()));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 250, 0, numOutputs);
		
		Runnable r = new Runnable() {
			@Override
			public void run() {
				try {
					log.info("Train model....");
					
					while(iter.hasNext()) {
						DataSet ds = iter.next();
						ds.normalizeZeroMeanZeroUnitVariance();
						getModel( 0 ).fit( ds ) ;
					}		
					log.info("Training done.");
					sil.getStream().offer( "Training done.");
				} catch( IllegalStateException er ) {
					log.error( "Error during training", er ) ;;
					sil.getStream().offer( er.getMessage() + "<br><br>Check number of inputs & outputs for compatability with your data." ) ;
				} finally {
					sil.getStream().offer( "" ) ;
				}
			}
		} ;
		new Thread( r ).start(); 
		return sil.getStream() ;
	}
	@Override
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
				.iterations(1000)
				//.useDropConnect(true)
				.learningRate(0.1)
				.regularization(true).l2(1e-4)
	            .list(numLayers)
	            ;
		
		for( int i=0 ; i<numLayers-1 ; i++ ) {
				lb.layer(i, new DenseLayer.Builder().nIn(numInputs).nOut( numInputs )
						.activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build()) ;
		}
		
		MultiLayerConfiguration conf = lb.layer(numLayers-1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.weightInit(WeightInit.XAVIER)
					.activation("softmax")
					.nIn(numInputs).nOut(numOutputs).build())				
			.backprop(true)
			.pretrain(false)
			.build();
		
		addModel(conf);  
	}
	
	

}

