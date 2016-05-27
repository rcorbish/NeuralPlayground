package com.rc;

import java.io.File;
import java.io.IOException;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultiLayer extends Model {

	private static Logger log = LoggerFactory.getLogger(Model.class);

	public MultiLayer(File trainingData, File testData, File configDir ) throws IOException {
		super( trainingData, testData, configDir ) ;
	}
	
	@Override
	public void train() throws Exception {

		getModel().setListeners(new ScoreIterationListener(100));

		log.info("Load data from " + trainingData );

		RecordReader recordReader = new CSVRecordReader(1);
		// Point to data path. 
		recordReader.initialize(new FileSplit(trainingData));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 200, 0, numOutputs);

		DataSet ds = null ;
		log.info("Train model....");
		while(iter.hasNext()) {
			ds = iter.next();
			ds.normalizeZeroMeanZeroUnitVariance();
			getModel().fit( ds ) ;
		}		
	}
	@Override
	public Evaluation<String> test() throws Exception {

		RecordReader recordReader = new CSVRecordReader(1);

		log.info("Load verification data from " + testData ) ;
		// Point to data path. 
		recordReader.initialize(new FileSplit(testData));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 200, 0, numOutputs );

		Evaluation<String> eval = new Evaluation<>( numOutputs );

		while(iter.hasNext()) {
			DataSet ds = iter.next();
			ds.normalizeZeroMeanZeroUnitVariance();
			INDArray predict2 = getModel().output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
			eval.eval(ds.getLabels(), predict2);
		}
		log.info(eval.stats());
		log.info("All Done");
		
		return eval ;
	}

	@Override
	public MultiLayerConfiguration createModelConfig() throws IOException {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(100)
				.useDropConnect(true)
				.learningRate(0.1)
				.regularization(true).l2(1e-4)
	            .list(4)
				.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numInputs+3)
						.activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
				.layer(1, new DenseLayer.Builder().nIn(numInputs+3).nOut(numInputs+5)
						.activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
				.layer(2, new DenseLayer.Builder().nIn(numInputs+5).nOut(numInputs+3)
						.activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.weightInit(WeightInit.XAVIER)
						.activation("softmax")
						.nIn(numInputs+3).nOut(numOutputs).build())
				.backprop(true)
				.pretrain(false)
				.build();
		return conf ; 
	}
}

