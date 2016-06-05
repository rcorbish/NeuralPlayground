package com.rc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.stream.Collectors;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTM extends Model {

	private static Logger log = LoggerFactory.getLogger(LSTM.class);

	public LSTM( Path configDir ) throws IOException {
		super( configDir ) ;
	}

	public LSTM() {
		super() ;
	}


	protected Collection<DataSet> getDatasets( Path source ) throws IOException {

		log.info("Load data from " + source );

		List<String[]> cols = Files.lines( source ) 
				.map( s -> s.split(",") )
				.collect( Collectors.toList() )
				;

		List<DataSet> dsets = new ArrayList<>() ;

		int timeSeriesLength = cols.get(0).length ;

		int numRowsProcessed = 0 ;
		while( numRowsProcessed<cols.size() ) {

			int batchSize = Math.min( 50, (cols.size()-numRowsProcessed) ) ;

			INDArray input  = Nd4j.zeros(batchSize, numInputs, timeSeriesLength ) ;
			INDArray labels = Nd4j.zeros(batchSize, numInputs, timeSeriesLength ) ;

			int ix[] = new int[3] ;
			for( int i=0 ; i<batchSize ; i++ ) {
				ix[0] = i ;
				String row[] = cols.get(numRowsProcessed) ;
				numRowsProcessed++ ;

				for( int j=1 ; j<timeSeriesLength ; j++ ) {
					ix[2] = j ;
					int v = row[j-1].charAt(0) - 'A' ;
					if( v>numInputs ) v = numInputs-1 ;
					if( v<0 ) v = 0 ;
					ix[1] = v ;
					input.putScalar( ix, 1.0 ) ;

					int w = row[j].charAt(0) - 'A' ;
					if( w>numInputs ) w = numInputs-1 ;
					if( w<0 ) w = 0 ;
					ix[1] = w ;					
					labels.putScalar( ix, 1.0 ) ;				
				}
			}
			dsets.add( new DataSet(input,labels) ) ;
		}
		return dsets ;
	}

	@Override
	public BlockingQueue<String> train( Path trainingData ) throws Exception {

		StreamIterationListener sil = new StreamIterationListener(100) ;
		forEach( mln -> mln.setListeners(sil) ) ;

		Collection<DataSet> dsets = getDatasets( trainingData ) ;

		Runnable r = new Runnable() {
			@Override
			public void run() {
				for( int epoch=0 ; epoch<100 ; epoch++ ) {
					log.info("Train model - epoch {}", epoch );

					for( DataSet ds : dsets ) { 
						getModel( 0 ).fit( ds ) ;
					}		
				}
				log.info("Training done.");
			}
		} ;
		new Thread( r ).start(); 
		return sil.getStream() ;
	}


	@Override
	public Evaluation test(Path testData) throws Exception {

		Collection<DataSet> dsets = getDatasets( testData ) ;

		Evaluation eval = new Evaluation();
		for( DataSet ds : dsets ) { 
			INDArray predict2 = getModel(0).output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
			eval.evalTimeSeries( ds.getFeatureMatrix(), predict2 ) ;
		}
		log.info(eval.stats());
		log.info("All Done");

		return eval ;
	}

	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		this.numInputs = numInputs ;
		this.numOutputs = numOutputs ;

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
				.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(numInputs).nOut(numInputs).build())				
				.backprop(true)
				.pretrain(false)
				.build();

		addModel(conf);  
	}
}

