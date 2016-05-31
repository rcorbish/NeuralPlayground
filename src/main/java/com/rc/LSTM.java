package com.rc;

import java.io.IOException;
import java.nio.file.Path;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTM extends Model {

	private static Logger log = LoggerFactory.getLogger(LSTM.class);

	public LSTM(Path trainingData, Path testData, Path configDir ) throws IOException {
		super( trainingData, testData, configDir ) ;
	}
	
	public LSTM() {
		super() ;
	}
	
	@Override
	public void train() throws Exception {

		forEach( mln -> mln.setListeners(new ScoreIterationListener(100) ) ) ;

		log.info("Load data from " + trainingData );

		RecordReader recordReader = new CSVRecordReader();
		// Point to data path. 
		recordReader.initialize(new FileSplit(trainingData.toFile()));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 100, 0, numOutputs);

		
		DataSet ds = null ;
		log.info("Train model....");
		
		while(iter.hasNext()) {
			ds = iter.next();
			ds.normalizeZeroMeanZeroUnitVariance();
			getModel( 0 ).fit( ds ) ;
		}		
		log.info("Training done.");
	}
	@Override
	public Evaluation test() throws Exception {

		RecordReader recordReader = new CSVRecordReader();

		log.info("Load verification data from " + testData ) ;
		// Point to data path. 
		recordReader.initialize(new FileSplit(testData.toFile()));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 100, 0, numOutputs );

		Evaluation eval = new Evaluation( numOutputs );
		while(iter.hasNext()) {
			DataSet ds = iter.next();
			ds.normalizeZeroMeanZeroUnitVariance();
			INDArray predict2 = getModel(0).output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
			eval.eval(ds.getFeatureMatrix(),predict2 );
		}
		log.info(eval.stats());
		log.info("All Done");
		
		return eval ;
	}

	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed( 100 )
				.iterations( 2 )
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(org.deeplearning4j.nn.conf.Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
	            .list(numLayers)
	            ;
		
		for( int i=0 ; i<numLayers-1 ; i++ ) {
				lb.layer(i, new GravesLSTM.Builder().nIn(numInputs).nOut(numInputs)
                        .activation("softsign").build()) ;
		}
		
		MultiLayerConfiguration conf = lb.layer(numLayers-1, new RnnOutputLayer.Builder().activation("softmax")
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(numInputs).nOut(numOutputs).build())				
			.backprop(true)
			.pretrain(false)
			.build();
		
		addModel(conf);  
	}
}

