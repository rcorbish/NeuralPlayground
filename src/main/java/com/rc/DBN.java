package com.rc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBN extends Model {
	private static Logger log = LoggerFactory.getLogger(DBN.class);


	public DataSet train( File dataFile ) throws Exception {
		final int numRows = 1;
		final int numColumns = 10;
		int seed = 123;
		int iterations = 1;
		int listenerFreq = iterations/5;

		getModel().setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

		log.info("Load data from " + dataFile );

		RecordReader recordReader = new CSVRecordReader(1);
		// Point to data path. 
		recordReader.initialize(new FileSplit(dataFile));
		WritableConverter converter = new WritableConverter() {
			@Override
			public Writable convert(Writable writable) throws WritableConverterException {
				return new IntWritable( writable.toString().equals("BIG") ? 1 : 0  ) ; 
			}
		};
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, converter, 100, 0, 2);

		log.info("Train model....");
		while(iter.hasNext()) {
			DataSet next = iter.next();
			getModel().fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
		}
		return null ;
	}


	public DataSet test( File dataFile ) throws Exception {

		Set<String> labels = new HashSet<>() ;
		try( BufferedReader br = new BufferedReader( new FileReader(dataFile)) ) {
			br.readLine() ;
			String s = br.readLine() ;
			while( s != null ) {
				int ix = s.indexOf(',') ;
				labels.add( s.substring(0,ix) ) ;
				s = br.readLine() ;
			}
			labels.remove(0) ;
		}
		
		RecordReader recordReader = new CSVRecordReader(1);

		log.info("Load verification data from " + dataFile ) ;
		WritableConverter converter = new WritableConverter() {
			@Override
			public Writable convert(Writable writable) throws WritableConverterException {
				return new IntWritable( writable.toString().equals("BIG") ? 1 : 0  ) ; 
			}
		};
		// Point to data path. 
		recordReader.initialize(new FileSplit(dataFile));
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, converter, 1, 0, labels.size() );
		
		Evaluation eval = new Evaluation( labels.size() );

		int n = 0 ;
		try {
			while(iter.hasNext()) {
				n++ ;
				DataSet ds = iter.next();
				INDArray predict2 = getModel().output(ds.getFeatureMatrix(), false);
				eval.eval(ds.getLabels(), predict2);
			}
		} catch( Throwable t ) {
			t.printStackTrace();
			log.info( "Managed to load " + n + " items" );
		}

		log.info(eval.stats());
		log.info("All Done");
		return null ;
	}

	@Override
	public MultiLayerConfiguration createModelConfig( File dataFile ) throws IOException {
		int numInputs = countInputsInDataFile(dataFile) ;
		int labelIndices[] = getLabelIndicesFromDataFile(dataFile) ;
		int numOutputs = countDistinctOutputsInDataFile(dataFile, labelIndices) ;

		final int numRows = 1;
		final int numColumns = 10;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(1000)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.list(10)
				.layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
				.layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
				.layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numRows*numColumns).build())
				.pretrain(true).backprop(true)
				.build();
		return conf ;
	}
}
